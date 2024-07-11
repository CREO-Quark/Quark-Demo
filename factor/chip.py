import json
from typing import Literal

import numpy as np
from algo_engine.base import MarketData
from algo_engine.base import TradeData, TransactionData
from quark.factor import ALPHA_01
from quark.factor import FactorMonitor, FixedIntervalSampler, AdaptiveVolumeIntervalSampler, Synthetic
from quark.factor.memory_core import NamedVector
from scipy.optimize import curve_fit
from scipy.stats import norm


class ChipDistributionMonitor(FactorMonitor, FixedIntervalSampler):

    def __init__(self, sampling_interval: float, sample_size: int, decay_factor: float = ALPHA_01,
                 name: str = 'Monitor.chip_distribution',
                 monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        FixedIntervalSampler.__init__(self=self, sampling_interval=sampling_interval, sample_size=sample_size)

        self.register_sampler(name='price', mode='update')
        self.register_sampler(name='volume', mode='accumulate')
        # self.register_sampler(name='volume_buy', mode='accumulate')
        # self.register_sampler(name='volume_sell', mode='accumulate')

        self.decay_factor = decay_factor
        self._chip_distribution: dict[str, dict[float, float]] = {}
        self._buy_distribution: dict[str, dict[float, float]] = {}
        self._sell_distribution: dict[str, dict[float, float]] = {}
        self._chip_params: dict[str, tuple[float, float, float, float, float]] = {}
        self.tick_size = 100  # todo: add docs here

    def on_entry_added(self, ticker: str, name: str, value):
        if name != 'price':
            return

        for idx in (chip_distribution := self._chip_distribution[ticker]):
            chip_distribution[idx] = chip_distribution[idx] * self.decay_factor

        for idx in (chip_distribution := self._buy_distribution[ticker]):
            chip_distribution[idx] = chip_distribution[idx] * self.decay_factor

        for idx in (chip_distribution := self._sell_distribution[ticker]):
            chip_distribution[idx] = chip_distribution[idx] * self.decay_factor

    def on_subscription(self, subscription: list[str] = None):
        super().on_subscription(subscription=subscription)

        for ticker in self.subscription:
            if ticker in self._chip_distribution:
                continue

            if not self.use_shm:
                self._chip_distribution[ticker] = {}
                self._buy_distribution[ticker] = {}
                self._sell_distribution[ticker] = {}
            else:
                self._chip_distribution[ticker]: NamedVector = self.memory_core.register(
                    name=f'chip_distribution.{ticker}',
                    dtype='NamedVector',
                    maxlen=self.sample_size)
                self._buy_distribution[ticker]: NamedVector = self.memory_core.register(
                    name=f'buy_distribution.{ticker}',
                    dtype='NamedVector',
                    maxlen=self.sample_size)
                self._sell_distribution[ticker]: NamedVector = self.memory_core.register(
                    name=f'sell_distribution.{ticker}',
                    dtype='NamedVector',
                    maxlen=self.sample_size)

    def update_chip_distribution(self, ticker: str, price: float, volume: float, side: Literal[-1, 1] | int = 0.) -> tuple[float, float]:

        if side == 1:
            chip_distribution = self._buy_distribution
        elif side == -1:
            chip_distribution = self._sell_distribution
        else:
            raise ValueError(f'Invalid side {side}.')

        if ticker in self._chip_distribution:
            chip_side = chip_distribution[ticker]
            chip_ttl = self._chip_distribution[ticker]
        elif not self.use_shm:
            chip_side = chip_distribution[ticker] = {}
            chip_ttl = self._chip_distribution[ticker] = {}
        else:
            raise KeyError(
                f'Ticker {ticker} must be registered before passing in {self.__class__.__name__}.update_chip_distribution!')

        idx = round(price * self.tick_size)
        chip_side[idx] = chip_side.get(idx, 0.) + volume * (1 - self.decay_factor)
        chip_ttl[idx] = chip_ttl.get(idx, 0.) + volume * (1 - self.decay_factor)

        return chip_side[idx], chip_ttl[idx]

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs) -> None:
        if isinstance(trade_data, (TradeData, TransactionData)):
            ticker = trade_data.ticker
            timestamp = trade_data.timestamp
            market_price = trade_data.market_price
            volume = trade_data.volume
            side = trade_data.side

            self.log_obs(ticker=ticker, timestamp=timestamp, price=market_price, volume=volume,
                         volume_buy=volume if side > 0 else 0, volume_sell=volume if side < 0 else 0)

            self.update_chip_distribution(ticker=ticker, price=market_price, volume=volume, side=side.sign)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')
        data_dict.update(
            chip_distribution={ticker: data for ticker, data in self._chip_distribution.items()}
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    def update_from_json(self, json_dict: dict):
        super().update_from_json(json_dict=json_dict)

        for ticker, data in json_dict['chip_distribution'].items():
            if ticker in self._chip_distribution:
                self._chip_distribution[ticker].update(data)
            elif not self.use_shm:
                self._chip_distribution[ticker] = {}
            else:
                self._chip_distribution[ticker]: NamedVector = self.memory_core.register(data,
                                                                                         name=f'chip_distribution.{ticker}',
                                                                                         dtype='NamedVector',
                                                                                         maxlen=self.sample_size)

        return self

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.{ticker}.chip' for ticker in subscription
        ] + [
            f'{self.name.removeprefix("Monitor.")}.{ticker}.buy' for ticker in subscription
        ] + [
            f'{self.name.removeprefix("Monitor.")}.{ticker}.sell' for ticker in subscription
        ]

    def dist_curve(self, x: np.ndarray, mu1: float, sigma1: float, weight1: float, mu2: float,
                   sigma2: float) -> np.ndarray:
        return weight1 * norm.pdf(x, mu1, sigma1) + (1 - weight1) * norm.pdf(x, mu2, sigma2)

    def fit_double_gaussian(self, ticker: str, chip: dict[float, float]):
        ttl_volume = np.sum(list(chip.values()))
        prices = np.array(list(chip.keys())) / self.tick_size
        volume_dist = np.array(list(chip.values())) / ttl_volume

        if ticker in self._chip_params:
            initial_params = self._chip_params[ticker]
        else:
            std = np.std(prices)
            mean1 = np.mean(prices) - std / 2
            weight1 = 0.5
            mean2 = np.mean(prices) + std / 2
            initial_params = [mean1, std, weight1, mean2, std]

        params, *_ = curve_fit(f=self.dist_curve, xdata=prices, ydata=volume_dist, p0=initial_params)
        self._chip_params[ticker] = params
        return params

    def _chip_indicator(self, chip_distribution: dict[str, dict[float, float]]) -> dict[str, float]:
        volume_dict = self.get_sampler(name='volume')
        price_dict = self.get_sampler(name='price')

        indicator = {}

        for ticker in volume_dict:
            # chip = self.update_chip_distribution(ticker, price_dict[ticker][-1], volume_dict[ticker][-1])
            chip = chip_distribution.get(ticker)

            if not chip:
                continue

            mu1, sigma1, weight1, mu2, sigma2 = params = self.fit_double_gaussian(ticker=ticker, chip=chip)
            dq_px = price_dict.get(ticker)

            if dq_px:
                px = dq_px[-1]
            else:
                continue

            if px < max(mu1, mu2) & px > min(mu1, mu2):
                pdf_derivative = (weight1 * norm.pdf(px, mu1, sigma1) * (px - mu1) / (sigma1 ** 2) +
                                  (1 - weight1) * norm.pdf(px, mu2, sigma2) * (px - mu2) / (sigma2 ** 2))
                min_pdf_derivative = -max(abs(pdf_derivative), 1)
                max_pdf_derivative = max(abs(pdf_derivative), 1)
                scaler = max(min_pdf_derivative, max_pdf_derivative)
                _indicator = pdf_derivative / scaler
                indicator[ticker] = _indicator
            elif px >= max(mu1, mu2):
                # _indicator = - (1-CDF(px))/(1-CDF(mu2)) + 1
                _indicator = - (1-(weight1 * norm.cdf(px, mu1, sigma1) + (1 - weight1) * norm.cdf(px, mu2, sigma2)))/(1-(weight1 * norm.cdf(mu2, mu1, sigma1) + (1 - weight1) * norm.cdf(mu2, mu2, sigma2))) + 1
                indicator[ticker] = _indicator
            else:
                # _indicator = - CDF(px)/CDF(mu1) - 1
                _indicator = (weight1 * norm.cdf(px, mu1, sigma1) + (1 - weight1) * norm.cdf(px, mu2, sigma2))/ (weight1 * norm.cdf(mu1, mu1, sigma1) + (1 - weight1) * norm.cdf(mu1, mu2, sigma2)) - 1
                indicator[ticker] = _indicator

        return indicator

    def calculate_scaled_pdf_derivative(self, params, current_price):
        mu1, sigma1, weight1, mu2, sigma2, weight2 = params
        pdf_derivative = (weight1 * norm.pdf(current_price, mu1, sigma1) * (current_price - mu1) / (sigma1 ** 2) +
                          weight2 * norm.pdf(current_price, mu2, sigma2) * (current_price - mu2) / (sigma2 ** 2))
        min_pdf_derivative = -max(abs(pdf_derivative), 1)
        max_pdf_derivative = max(abs(pdf_derivative), 1)
        scaler = max(min_pdf_derivative, max_pdf_derivative)
        return pdf_derivative / scaler

    @property
    def value(self) -> dict[str, float]:

        indicator = {}

        # 打印调试信息
        chip_indicator = self._chip_indicator(self._chip_distribution)
        chip_indicator_buy = self._chip_indicator(self._buy_distribution)
        chip_indicator_sell = self._chip_indicator(self._sell_distribution)
        print("Chip Indicator: ", chip_indicator)
        print("Chip Indicator Buy: ", chip_indicator_buy)
        print("Chip Indicator Sell: ", chip_indicator_sell)

        indicator.update({f'{ticker}.chip': _indicator
                          for ticker, _indicator in self._chip_indicator(self._chip_distribution).items()
                          })

        indicator.update({f'{ticker}.chip.buy': _indicator
                          for ticker, _indicator in self._chip_indicator(self._buy_distribution).items()
                          })

        indicator.update({f'{ticker}.chip.sell': _indicator
                          for ticker, _indicator in self._chip_indicator(self._sell_distribution).items()
                          })

        return indicator


class ChipDistributionAdaptiveMonitor(ChipDistributionMonitor, AdaptiveVolumeIntervalSampler):

    def __init__(self, sampling_interval: float, sample_size: int = 20, baseline_window: int = 100,
                 aligned_interval: bool = False, name: str = 'Monitor.chip_distribution.Adaptive',
                 monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            name=name,
            monitor_id=monitor_id
        )

        AdaptiveVolumeIntervalSampler.__init__(
            self=self,
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            baseline_window=baseline_window,
            aligned_interval=aligned_interval
        )

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs):
        self.accumulate_volume(market_data=trade_data)
        super().on_trade_data(trade_data=trade_data, **kwargs)

    @property
    def is_ready(self) -> bool:
        return self.baseline_ready


class ChipAdaptiveIndexMonitor(ChipDistributionAdaptiveMonitor, Synthetic):

    def __init__(self, sampling_interval: float, sample_size: int, baseline_window: int, aligned_interval: bool = False,
                 weights: dict[str, float] = None, name: str = 'Monitor.chip_distribution.Adaptive.Index',
                 monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            baseline_window=baseline_window,
            aligned_interval=aligned_interval,
            name=name,
            monitor_id=monitor_id
        )

        Synthetic.__init__(self=self, weights=weights)

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.chip_indicator',
            f'{self.name.removeprefix("Monitor.")}.chip_indicator_buy',
            f'{self.name.removeprefix("Monitor.")}.chip_indicator_sell'
        ]

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker

        if self.weights and ticker not in self.weights:
            return

        super().__call__(market_data=market_data, **kwargs)

    @property
    def value(self) -> dict[str, float]:
        chip_indicator = self._chip_indicator(self._chip_distribution)
        chip_indicator_buy = self._chip_indicator(self._buy_distribution)
        chip_indicator_sell = self._chip_indicator(self._sell_distribution)

        return {
            'chip_indicator': self.composite(values=chip_indicator),
            'chip_indicator_buy': self.composite(values=chip_indicator_buy),
            'chip_indicator_sell': self.composite(values=chip_indicator_sell)
        }
