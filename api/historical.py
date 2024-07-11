import csv
import datetime
import os.path
import pathlib
import sys
import time
from collections.abc import Iterable
from functools import wraps
from io import BytesIO

import numpy as np
import py7zr
from algo_engine.base import TradeData, TransactionData
from quark.base import GlobalStatics

from . import LOGGER

TIME_ZONE = GlobalStatics.TIME_ZONE
DEBUG_MODE = GlobalStatics.DEBUG_MODE


class SMBClient(object):
    def __init__(self, **kwargs):
        self.username = kwargs.get('username', 'creo')
        self.password = kwargs.get('password', 'creo')
        self.smb_host = kwargs.get('smb_host', '192.168.3.10')
        self.smb_name = kwargs.get('smb_name', 'FileServer')
        self.smb_root = kwargs.get('smb_root', 'FileServer')
        self.conn = None

    def __del__(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    @staticmethod
    def connect(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.conn is None:
                from smb.SMBConnection import SMBConnection
                self.conn = SMBConnection(username=self.username, password=self.password, my_name='Py_SMB_API', remote_name=self.smb_name)
                assert self.conn.connect(self.smb_host)
                LOGGER.info(f'SMB {self.smb_name} Connected!\n' + '\n'.join([f'\t- {share.name}: {share.comments}' for share in self.conn.listShares()]))
            r = func(self, *args, **kwargs)

            self.conn.close()
            return r

        return wrapper

    @connect
    def read(self, file_path: str | pathlib.Path) -> BytesIO:
        file_obj = BytesIO()
        try:
            LOGGER.info(f'Retrieving file {file_path}...')
            self.conn.retrieveFile(service_name=self.smb_root, path=file_path, file_obj=file_obj)
            LOGGER.info(f'{file_path} retrieved successfully! File size {file_obj.getbuffer().nbytes:,} bytes')
            file_obj.seek(0)
        except Exception as e:
            raise FileNotFoundError(f'{file_path} not found on SMB server!') from e
        return file_obj


class ArchiveReader(object):
    def __init__(self, **kwargs):
        self.archive_dir = kwargs.get('archive_dir')
        self.extract_dir = kwargs.get('extract_dir', pathlib.Path.home().joinpath('Documents', 'TradeData'))

        self.smb = None

        if self.archive_dir is None:
            # check if archives located in local documents
            if os.path.isdir(archive_dir := pathlib.Path.home().joinpath('Documents', 'TradeDataArchive')):
                self.archive_dir = archive_dir
            # check if the archive dir is mounted
            elif os.path.isdir(archive_dir := pathlib.Path('/mnt', 'FileServer', 'shared', 'TradeDataArchive')):
                self.archive_dir = archive_dir
            # fallback to use smb server
            else:
                self.archive_dir = 'smb://192.168.3.10/FileServer/shared'
                self.smb = SMBClient(username='creo', password='creo', smb_host='192.168.3.10', smb_name='FileServer')

    def unzip_local(self, market_date: datetime.date, ticker: str):
        archive_path = pathlib.Path(self.archive_dir, f'{market_date:%Y%m}', f'{market_date:%Y-%m-%d}.7z')
        destination_path = pathlib.Path(self.extract_dir)
        directory_to_extract = f'{market_date:%Y-%m-%d}'
        file_to_extract = f'{ticker.split(".")[0]}.csv'

        if not os.path.isfile(archive_path):
            raise FileNotFoundError(f'{archive_path} not found!')

        os.makedirs(destination_path, exist_ok=True)

        LOGGER.info(f'Unzipping {file_to_extract} from {archive_path} to {destination_path}...')

        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            archive.extract(targets=[f'{directory_to_extract}/{file_to_extract}'], path=destination_path)

        return 0

    def unzip_batch_local(self, market_date: datetime.date, ticker_list: Iterable[str]):
        archive_path = pathlib.Path(self.archive_dir, f'{market_date:%Y%m}', f'{market_date:%Y-%m-%d}.7z')
        destination_path = pathlib.Path(self.extract_dir)
        directory_to_extract = f'{market_date:%Y-%m-%d}'

        targets = []

        for ticker in ticker_list:
            name = f'{ticker.split(".")[0]}.csv'
            destination = pathlib.Path(self.extract_dir, f'{market_date:%Y-%m-%d}', f'{ticker.split(".")[0]}.csv')

            if os.path.isfile(destination):
                continue

            targets.append(f'{directory_to_extract}/{name}')

        if not os.path.isfile(archive_path):
            raise FileNotFoundError(f'{archive_path} not found!')

        os.makedirs(destination_path, exist_ok=True)

        if not targets:
            return 0

        LOGGER.info(f'Unzipping {len(targets)} names from {archive_path} to {destination_path}...')

        with py7zr.SevenZipFile(archive_path, mode='r') as archive:
            archive.extract(targets=targets, path=destination_path)

        return 0

    def unzip_smb(self, market_date: datetime, ticker: str):
        smb_archive_path = f'TradeDataArchive/{market_date:%Y%m}/{market_date:%Y-%m-%d}.7z'
        destination_path = pathlib.Path(self.extract_dir)
        directory_to_extract = f'{market_date:%Y-%m-%d}'
        file_to_extract = f'{ticker.split(".")[0]}.csv'

        file_obj = self.smb.read(smb_archive_path)

        os.makedirs(destination_path, exist_ok=True)

        LOGGER.info(f'Unzipping {file_to_extract} from SMB archive {smb_archive_path} to {destination_path}...')

        with py7zr.SevenZipFile(file_obj, mode='r') as archive:
            archive.extract(targets=[f'{directory_to_extract}/{file_to_extract}'], path=destination_path)

        return 0

    def unzip_batch_smb(self, market_date: datetime.date, ticker_list: Iterable[str]):
        import py7zr

        smb_archive_path = f'TradeDataArchive/{market_date:%Y%m}/{market_date:%Y-%m-%d}.7z'
        destination_path = pathlib.Path(self.extract_dir)
        directory_to_extract = f'{market_date:%Y-%m-%d}'

        targets = []

        for ticker in ticker_list:
            name = f'{ticker.split(".")[0]}.csv'
            destination = pathlib.Path(self.extract_dir, f'{market_date:%Y-%m-%d}', f'{ticker.split(".")[0]}.csv')

            if os.path.isfile(destination):
                continue

            targets.append(f'{directory_to_extract}/{name}')

        if not targets:
            return 0

        os.makedirs(destination_path, exist_ok=True)
        file_obj = self.smb.read(smb_archive_path)

        LOGGER.info(f'Unzipping {len(targets)} names from SMB archive {pathlib.Path(self.archive_dir, smb_archive_path)} to {destination_path}...')
        with py7zr.SevenZipFile(file_obj, mode='r') as archive:
            archive.extract(targets=targets, path=destination_path)

        return 0

    def unzip(self, market_date: datetime.date, ticker: str):
        if self.archive_dir.startswith('smb://'):
            self.unzip_smb(market_date=market_date, ticker=ticker)
        else:
            self.unzip_local(market_date=market_date, ticker=ticker)

    def unzip_batch(self, market_date: datetime.date, ticker_list: Iterable[str]):
        if self.archive_dir.startswith('smb://'):
            self.unzip_batch_smb(market_date=market_date, ticker_list=ticker_list)
        else:
            self.unzip_batch_local(market_date=market_date, ticker_list=ticker_list)

    def load_trade_data(self, market_date: datetime.date, ticker: str, dtype: str) -> list[TradeData]:
        ts = time.time()
        trade_data_list = []

        file_path = pathlib.Path(self.extract_dir, f'{market_date:%Y-%m-%d}', f'{ticker.split(".")[0]}.csv')

        if not os.path.isfile(file_path):
            try:
                self.unzip(market_date=market_date, ticker=ticker)
            except FileNotFoundError as _:
                return trade_data_list

        with open(file_path, 'r') as f:
            data_file = csv.DictReader(f)
            for row in data_file:  # type: dict[str, str | float]

                if dtype == 'TradeData':
                    constractor = TradeData
                    if row['Type'] == 0:
                        continue
                elif dtype == 'TransactionData':
                    constractor = TransactionData
                else:
                    raise ValueError(f'Invalid dtype {dtype}!')

                trade_data = constractor(
                    ticker=ticker,
                    price=float(row['Price']),
                    volume=float(row['Volume']),
                    timestamp=datetime.datetime.combine(market_date, datetime.time(*map(int, row['Time'].split(":"))), TIME_ZONE).timestamp(),
                    side=row['Type'],
                    buy_id=int(row['BuyOrderID']),
                    sell_id=int(row['SaleOrderID'])
                )
                trade_data_list.append(trade_data)

                if DEBUG_MODE:
                    if not np.isfinite(trade_data.volume):
                        raise ValueError(f'Invalid trade data {trade_data}, volume = {trade_data.volume}')

                    if not np.isfinite(trade_data.price) or trade_data.price < 0:
                        raise ValueError(f'Invalid trade data {trade_data}, price = {trade_data.price}')

                    if trade_data.side.value == 0:
                        raise ValueError(f'Invalid trade data {trade_data}, side = {trade_data.side}')

        LOGGER.info(f'{market_date} {ticker} trade data loaded, {len(trade_data_list):,} entries in {time.time() - ts:.3f}s.')

        return trade_data_list

    def __call__(self, market_date: datetime.date, ticker: str, dtype: str):
        if dtype in ('TradeData', 'TransactionData'):
            return self.load_trade_data(market_date=market_date, ticker=ticker, dtype=dtype)
        else:
            raise NotImplementedError(f'API.historical does not have a loader function for {dtype}')


def loader(market_date: datetime.date, ticker: str, dtype: str):
    reader = ArchiveReader()
    return reader(market_date=market_date, ticker=ticker, dtype=dtype)


def unzip(market_date: datetime.date, ticker: str):
    reader = ArchiveReader()
    return reader.unzip(market_date=market_date, ticker=ticker)


def unzip_batch(market_date: datetime.date, ticker_list: Iterable[str]):
    reader = ArchiveReader()
    return reader.unzip_batch(market_date=market_date, ticker_list=ticker_list)


def main():
    market_date = datetime.date(2023, 1, 3)
    reader = ArchiveReader()

    reader.unzip_batch_smb(market_date=market_date, ticker_list=['000016.SZ', '123.asd'])
    r = reader.__call__(market_date=market_date, ticker='000016.SZ', dtype='TradeData')
    # print(r)


if __name__ == '__main__':
    main()

    LOGGER.info('All done!')
    sys.exit(0)
