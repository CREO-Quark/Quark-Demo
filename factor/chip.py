import json
from typing import Literal

import numpy as np
from algo_engine.base import MarketData
from algo_engine.base import TradeData, TransactionData
from quark.factor import ALPHA_001, ALPHA_01
from quark.factor import FactorMonitor, FixedIntervalSampler, AdaptiveVolumeIntervalSampler, Synthetic, EMA, SamplerMode
from quark.factor import LOGGER
from quark.factor.memory_core import NamedVector
from scipy.optimize import minimize
from scipy.stats import norm


class ChipDistributionMonitor(FactorMonitor, FixedIntervalSampler):

    def __init__(self, sampling_interval: float, sample_size: int, decay_factor: float = ALPHA_001,
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
        self._chip_params: dict[str, tuple[float, float, float, float, float] | list[float]] = {}
        self._buy_params: dict[str, tuple[float, float, float, float, float] | list[float]] = {}
        self._sell_params: dict[str, tuple[float, float, float, float, float] | list[float]] = {}
        self.tick_size = 100
        self.calibration_timestamp: dict[str, float] = {}
        self.calibration_timestamp_buy: dict[str, float] = {}
        self.calibration_timestamp_sell: dict[str, float] = {}
        self.calibration_interval = 60

        self.timestamp = 0.

    def on_entry_added(self, ticker: str, name: str, value):
        if name != 'price':
            return

        if not self.use_shm:
            if ticker in self._chip_distribution:
                chip_distribution = self._chip_distribution[ticker]
            else:
                chip_distribution = self._chip_distribution[ticker] = {}

            if ticker in self._buy_distribution:
                buy_chip_distribution = self._buy_distribution[ticker]
            else:
                buy_chip_distribution = self._buy_distribution[ticker] = {}

            if ticker in self._sell_distribution:
                sell_chip_distribution = self._sell_distribution[ticker]
            else:
                sell_chip_distribution = self._sell_distribution[ticker] = {}
        else:
            chip_distribution = self._chip_distribution[ticker]
            buy_chip_distribution = self._buy_distribution[ticker]
            sell_chip_distribution = self._sell_distribution[ticker]

        for idx in chip_distribution:
            chip_distribution[idx] = chip_distribution[idx] * self.decay_factor

        for idx in buy_chip_distribution:
            buy_chip_distribution[idx] = buy_chip_distribution[idx] * self.decay_factor

        for idx in sell_chip_distribution:
            sell_chip_distribution[idx] = sell_chip_distribution[idx] * self.decay_factor

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
        chip_side[idx] = chip_side.get(idx, 0.) + volume
        chip_ttl[idx] = chip_ttl.get(idx, 0.) + volume

        return chip_side[idx], chip_ttl[idx]

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs) -> None:
        if isinstance(trade_data, (TradeData, TransactionData)):
            ticker = trade_data.ticker
            timestamp = trade_data.timestamp
            market_price = trade_data.market_price
            volume = trade_data.volume
            side = trade_data.side

            self.log_obs(ticker=ticker, timestamp=timestamp, price=market_price, volume=volume)
            self.update_chip_distribution(ticker=ticker, price=market_price, volume=volume, side=side.sign)
        self.timestamp = trade_data.timestamp

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
        ] + [
            f'{self.name.removeprefix("Monitor.")}.{ticker}.diff' for ticker in subscription
        ]

    # def dist_curve(self, x: np.ndarray, mu1: float, sigma1: float, weight1: float, mu2: float,
    #                sigma2: float) -> np.ndarray:
    #     return weight1 * norm.pdf(x, mu1, sigma1) + (1 - weight1) * norm.pdf(x, mu2, sigma2)

    # def fit_double_gaussian(self, ticker: str, chip: dict[float, float]):
    #     # todo: check your regression model
    #     ttl_volume = np.sum(list(chip.values()))
    #     prices = np.array(list(chip.keys())) / self.tick_size
    #     volume_dist = np.array(list(chip.values())) / ttl_volume
    #
    #     if ticker in self._chip_params:
    #         initial_params = self._chip_params[ticker]
    #     else:
    #         std = np.std(prices)
    #         mean1 = np.mean(prices) - std / 2
    #         weight1 = 0.5
    #         mean2 = np.mean(prices) + std / 2
    #         initial_params = [mean1, std, weight1, mean2, std]
    #
    #     params, *_ = curve_fit(f=self.dist_curve, xdata=prices, ydata=volume_dist, p0=initial_params)
    #     self._chip_params[ticker] = params
    #     return params

    def neg_log_likelihood(self, params, x, y):
        mean1, std1, weight1, mean2, std2 = params
        if std1 <= 0 or std2 <= 0 or weight1 < 0 or weight1 > 1:
            return np.inf
        pdf1 = norm.pdf(x, mean1, std1)
        pdf2 = norm.pdf(x, mean2, std2)
        mixture_pdf = weight1 * pdf1 + (1 - weight1) * pdf2

        # 防止对数零值
        eps = 1e-10
        mixture_pdf = np.clip(mixture_pdf, eps, None)

        nll = -np.sum(y * np.log(mixture_pdf))
        return nll

    def fit_double_gaussian(self, chip: dict[float, float], init_params: tuple[float, ...]) -> list[float]:
        ttl_volume = np.sum(list(chip.values()))
        prices = np.array(list(chip.keys())) / self.tick_size
        volume_dist = np.array(list(chip.values())) / ttl_volume

        if init_params:
            initial_params = init_params
        else:
            std = np.std(prices)
            mean1 = np.mean(prices) - std / 2
            weight1 = 0.5
            mean2 = np.mean(prices) + std / 2
            initial_params = [mean1, std, weight1, mean2, std]

        # 使用 minimize 函数进行优化
        result = minimize(self.neg_log_likelihood,
                          np.array(initial_params),
                          args=(prices, volume_dist),
                          bounds=[(None, None), (0.001, None), (0, 1), (None, None), (0.001, None)])
        params = result.x
        return list(params)

    def _chip_indicator(self, chip_distribution: dict[str, dict[float, float]], chip_params: dict[str, list[float]] = None, calibration_timestamp: dict[str, float] = None) -> dict[str, float]:
        price_dict = self.get_sampler(name='price')

        if chip_params is None:
            if chip_distribution is self._chip_distribution:
                chip_params = self._chip_params
            elif chip_distribution is self._buy_distribution:
                chip_params = self._buy_params
            elif chip_distribution is self._sell_distribution:
                chip_params = self._sell_params
            else:
                LOGGER.warning('No chip distribution provided.')
                chip_params = self._chip_params

        if calibration_timestamp is None:
            if chip_distribution is self._chip_distribution:
                calibration_timestamp = self.calibration_timestamp
            elif chip_distribution is self._buy_distribution:
                calibration_timestamp = self.calibration_timestamp_buy
            elif chip_distribution is self._sell_distribution:
                calibration_timestamp = self.calibration_timestamp_sell
            else:
                LOGGER.warning('No chip distribution provided.')
                calibration_timestamp = self.calibration_timestamp

        indicator = {}

        for ticker in chip_distribution:
            # chip = self.update_chip_distribution(ticker, price_dict[ticker][-1], volume_dict[ticker][-1])
            chip = chip_distribution.get(ticker)
            last_calibration_timestamp = calibration_timestamp.get(ticker, 0.)

            if not chip:
                continue
            if not last_calibration_timestamp or self.timestamp - last_calibration_timestamp >= self.calibration_interval:
                mu1, sigma1, weight1, mu2, sigma2 = params = self.fit_double_gaussian(chip=chip, init_params=chip_params.get(ticker))
                chip_params[ticker] = params
                calibration_timestamp[ticker] = self.timestamp
            else:
                if ticker in chip_params:
                    mu1, sigma1, weight1, mu2, sigma2 = chip_params[ticker]
                else:
                    mu1, sigma1, weight1, mu2, sigma2 = params = self.fit_double_gaussian(chip=chip, init_params=chip_params.get(ticker))
                    chip_params[ticker] = params
                    calibration_timestamp[ticker] = self.timestamp

            dq_px = price_dict.get(ticker)

            if dq_px:
                px = dq_px[-1]
            else:
                continue

            if min(mu1, mu2) < px < max(mu1, mu2):
                pdf_derivative = (weight1 * norm.pdf(px, mu1, sigma1) * (px - mu1) / (sigma1 ** 2) +
                                  (1 - weight1) * norm.pdf(px, mu2, sigma2) * (px - mu2) / (sigma2 ** 2))
                min_pdf_derivative = -max(abs(pdf_derivative), 1)
                max_pdf_derivative = max(abs(pdf_derivative), 1)
                scaler = max(min_pdf_derivative, max_pdf_derivative)
                _indicator = pdf_derivative / scaler
                indicator[ticker] = _indicator
            elif px >= max(mu1, mu2):
                # _indicator = - (1-CDF(px))/(1-CDF(mu2)) + 1
                _indicator = - (
                        1 - (weight1 * norm.cdf(px, mu1, sigma1) + (1 - weight1) * norm.cdf(px, mu2, sigma2))) / (
                                     1 - (weight1 * norm.cdf(mu2, mu1, sigma1) + (1 - weight1) * norm.cdf(mu2, mu2,
                                                                                                          sigma2))) + 1
                indicator[ticker] = _indicator
            else:
                # _indicator = - CDF(px)/CDF(mu1) - 1
                _indicator = (weight1 * norm.cdf(px, mu1, sigma1) + (1 - weight1) * norm.cdf(px, mu2, sigma2)) / (
                        weight1 * norm.cdf(mu1, mu1, sigma1) + (1 - weight1) * norm.cdf(mu1, mu2, sigma2)) - 1
                indicator[ticker] = _indicator

        return indicator

    def _get_indicator(self, ticker: str | list[str] = None) -> dict[str, float]:
        indicator = {}
        if ticker is None:
            chip_indicator = self._chip_indicator(self._chip_distribution)
            chip_indicator_buy = self._chip_indicator(self._buy_distribution)
            chip_indicator_sell = self._chip_indicator(self._sell_distribution)
        else:
            if isinstance(ticker, str):
                ticker = [ticker]
            chip_indicator = self._chip_indicator({_ticker: self._chip_distribution[_ticker] for _ticker in ticker}, chip_params=self._chip_params, calibration_timestamp=self.calibration_timestamp)
            chip_indicator_buy = self._chip_indicator({_ticker: self._buy_distribution[_ticker] for _ticker in ticker}, chip_params=self._buy_params, calibration_timestamp=self.calibration_timestamp_buy)
            chip_indicator_sell = self._chip_indicator({_ticker: self._sell_distribution[_ticker] for _ticker in ticker}, chip_params=self._sell_params, calibration_timestamp=self.calibration_timestamp_sell)

        chip_indicator_diff = {ticker: chip_indicator_buy[ticker] - chip_indicator_sell[ticker] for ticker in set(chip_indicator_buy) & set(chip_indicator_sell)}

        # print("Chip Indicator: ", chip_indicator)
        # print("Chip Indicator Buy: ", chip_indicator_buy)
        # print("Chip Indicator Sell: ", chip_indicator_sell)
        # print("Chip Indicator Diff: ", chip_indicator_diff)

        indicator.update(
            **{f'{ticker}.chip': _indicator for ticker, _indicator in chip_indicator.items()},
            **{f'{ticker}.chip.buy': _indicator for ticker, _indicator in chip_indicator_buy.items()},
            **{f'{ticker}.chip.sell': _indicator for ticker, _indicator in chip_indicator_sell.items()},
            **{f'{ticker}.chip.diff': _indicator for ticker, _indicator in chip_indicator_diff.items()}
        )

        return indicator

    @property
    def value(self) -> dict[str, float]:
        indicator = self._get_indicator()
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

        self.ema = EMA(alpha=ALPHA_01)
        self.ema.register_ema(name='chip')
        self.chip_ema = self.ema.ema['chip']
        self.register_sampler(name='amplitude', mode=SamplerMode.accumulate)

    def on_entry_added(self, ticker: str, name: str, value):
        super().on_entry_added(ticker=ticker, name=name, value=value)

        if name == 'price':
            chip_indicator = self._chip_indicator(self._chip_distribution)
            if ticker in chip_indicator:
                self.ema.update_ema(ticker=ticker, chip=chip_indicator[ticker], replace_na=0.)
                self.log_obs(ticker=ticker, amplitude=abs(chip_indicator[ticker] - self.chip_ema[ticker]), timestamp=self.timestamp)

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.chip_indicator',
            f'{self.name.removeprefix("Monitor.")}.chip_indicator_buy',
            f'{self.name.removeprefix("Monitor.")}.chip_indicator_sell',
            f'{self.name.removeprefix("Monitor.")}.chip_indicator_diff_amplitude'
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
        # chip_indicator_diff = {ticker: chip_indicator_buy[ticker] - chip_indicator_sell[ticker] for ticker in set(chip_indicator_buy) & set(chip_indicator_sell)}
        chip_indicator_ema_diff = {ticker: chip_indicator[ticker] - self.chip_ema[ticker] for ticker in self.chip_ema}
        chip_diff_amplitude = {ticker: sum(list(dq)[-15:]) for ticker, dq in self.get_sampler(name='amplitude').items()}

        return {
            'chip_indicator': self.composite(values=chip_indicator),
            'chip_indicator_buy': self.composite(values=chip_indicator_buy),
            'chip_indicator_sell': self.composite(values=chip_indicator_sell),
            'chip_indicator_diff_amplitude': self.composite(values=chip_diff_amplitude)
        }
