import csv
import datetime
import hashlib
import os.path
import pathlib
import re
import shutil
import subprocess
import sys
import traceback
from functools import wraps
from io import StringIO

import pandas as pd
import psycopg2
import sqlalchemy
import sqlalchemy.exc
from algo_engine.base import TickData, TradeData, TransactionData
from quark.base import LOGGER


class PostgresClient(object):
    def __init__(self, **kwargs):
        self.db_name = kwargs.get('db_name', 'postgres')
        self.db_address = kwargs.get('db_address', '192.168.2.11')
        self.db_port = kwargs.get('db_port', 5432)
        self.db_user = kwargs.get('db_user', 'postgres')
        self.db_password = kwargs.get('db_password', 'Tze1LewdF6WEN*1M@3uHYIjIpzFwDXTS')
        self.db_conn = None

        # Database connection string
        self.db_url = sqlalchemy.engine.url.URL.create(drivername='postgresql+psycopg2', username=self.db_user, password=self.db_password, host=self.db_address, port=self.db_port, database=self.db_name)
        self.engine = sqlalchemy.create_engine(self.db_url)

    @staticmethod
    def connect(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.db_conn is None or self.db_conn.closed:
                self.db_conn = psycopg2.connect(
                    dbname=self.db_name,
                    user=self.db_user,
                    password=self.db_password,
                    host=self.db_address,
                    port=self.db_port
                )

            r = func(self, *args, **kwargs)

            self.db_conn.close()
            return r

        return wrapper

    @connect
    def create_database(self, db_name: str = None, exist_ok: bool = True):
        if db_name is None:
            db_name = self.db_name

        # Connect to the default database to create the new database
        conn = self.db_conn
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()

        if not exists:
            cursor.execute(f"CREATE DATABASE {db_name}")
        elif not exist_ok:
            raise ValueError(f'{db_name} existed!')

        cursor.close()
        conn.close()

    @connect
    def create_schema(self, schema_name: str, db_name: str = None):
        conn = self.db_conn

        with conn.cursor() as cursor:
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}";')

        conn.commit()

    @connect
    def delete_empty_schemas(self):
        conn = self.db_conn

        with conn.cursor() as cursor:
            # Fetch all schemas
            cursor.execute("SELECT schema_name FROM information_schema.schemata;")
            schemas = cursor.fetchall()
            date_pattern = re.compile(r'^\d{8}$')  # Pattern to match YYYYMMDD

            for schema in schemas:
                schema_name = schema[0]

                if not date_pattern.match(schema_name):
                    continue

                # Check if the schema has any tables
                cursor.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{schema_name}';")
                table_count = cursor.fetchone()[0]

                # If there are no tables, drop the schema
                if table_count == 0:
                    cursor.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE;')
                    LOGGER.info(f'Schema {schema_name} dropped.')

        conn.commit()

    @connect
    def check_exists(self, schema: str, table: str) -> bool:
        conn = self.db_conn
        exists = False

        with conn.cursor() as cursor:
            # Check if the schema exists
            cursor.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name = %s;", (schema,))
            schema_exists = cursor.fetchone() is not None

            if schema_exists:
                # Check if the table exists in the schema
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = %s AND table_name = %s;",
                    (schema, table)
                )
                table_exists = cursor.fetchone() is not None
                exists = table_exists

        return exists

    def upload_df(self, data: pd.DataFrame, table_name: str, schema: str = None, **kwargs):
        data.to_sql(
            name=table_name,
            con=self.engine,
            schema=schema,
            **kwargs
        )
        LOGGER.info(f'{table_name} uploaded!')

    @classmethod
    def psql_insert_copy(cls, table, conn, keys, data_iter):
        """
        Execute SQL statement inserting data

        Parameters
        ----------
        table : pandas.io.sql.SQLTable
        conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
        keys : list of str
            Column names
        data_iter : Iterable that iterates the values to be inserted
        """
        # gets a DBAPI connection that can provide a cursor
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            s_buf = StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)

            columns = ', '.join(['"{}"'.format(k) for k in keys])
            if table.schema:
                table_name = '{}.{}'.format(table.schema, table.name)
            else:
                table_name = table.name

            sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(table_name, columns)
            cur.copy_expert(sql=sql, file=s_buf)

    def has_table(self, table_name: str, schema: str = None):
        inspector = sqlalchemy.inspect(self.engine)
        return inspector.has_table(table_name=table_name, schema=schema)


class FutureTickUploader(PostgresClient):
    column_mapping = {
        '最新价': 'last_price',
        '上次结算价': 'prev_settlement_price',
        '昨收盘': 'prev_close_price',
        '昨持仓量': 'prev_open_interest',
        '今开盘': 'open_price',
        '最高价': 'highest_price',
        '最低价': 'lowest_price',
        '数量': 'volume',
        '成交金额': 'turnover',
        '持仓量': 'open_interest',
        '今收盘': 'close_price',
        '本次结算价': 'current_settlement_price',
        '涨停板价': 'upper_limit_price',
        '跌停板价': 'lower_limit_price',
        '昨虚实度': 'prev_hedge_ratio',
        '今虚实度': 'current_hedge_ratio',
        '申买价一': 'bid_price_1',
        '申买量一': 'bid_volume_1',
        '申卖价一': 'ask_price_1',
        '申卖量一': 'ask_volume_1',
        '申买价二': 'bid_price_2',
        '申买量二': 'bid_volume_2',
        '申卖价二': 'ask_price_2',
        '申卖量二': 'ask_volume_2',
        '申买价三': 'bid_price_3',
        '申买量三': 'bid_volume_3',
        '申卖价三': 'ask_price_3',
        '申卖量三': 'ask_volume_3',
        '申买价四': 'bid_price_4',
        '申买量四': 'bid_volume_4',
        '申卖价四': 'ask_price_4',
        '申卖量四': 'ask_volume_4',
        '申买价五': 'bid_price_5',
        '申买量五': 'bid_volume_5',
        '申卖价五': 'ask_price_5',
        '申卖量五': 'ask_volume_5',
        '当日均价': 'average_price',
        '业务日期': 'business_date'
    }

    filename_mapping = {
        '次主力连续': '_second_main',
        '主力连续': '_main',
        '次_main': '_second_main',
        '下月连续': '_next_month',
        '下季连续': '_next_season',
        '当月连续': '_current_month',
        '当季连续': '_current_season',
        '隔季连续': '_season_after_next'
    }

    def __init__(self, start_date: datetime, end_date: datetime.date, **kwargs):
        self.start_date = start_date
        self.end_date = end_date

        super().__init__(db_name='stock_future', **kwargs)

        self.data_dir = pathlib.Path.home().joinpath('Documents', 'StockFutureTickData')

    def _rename_files(self, data_dir: pathlib.Path):
        for file_name in os.listdir(data_dir):
            new_file_name = file_name

            for chinese, english in self.filename_mapping.items():
                new_file_name = new_file_name.replace(chinese, english)

            if new_file_name != file_name:
                old_file = data_dir.joinpath(file_name)
                new_file = data_dir.joinpath(new_file_name)
                os.rename(old_file, new_file)

    def _hash_and_link_files(self, data_dir: str | pathlib.Path):
        data_dir = pathlib.Path(data_dir)  # Ensure data_dir is a Path object
        hash_mapping = {}
        file_links = {}

        # Generate mapping with actual ticker files
        for file_name in os.listdir(data_dir):
            if not file_name.split('_')[0][-1].isdigit():
                continue

            market_file = data_dir.joinpath(file_name)
            file_hash = self._hash(file_path=market_file)
            if file_hash in hash_mapping:
                raise ValueError(f'Hash collision {file_name} with {hash_mapping[file_hash]}')
            hash_mapping[file_hash] = file_name

        # Link the logic ticker files with actual ticker files
        for file_name in os.listdir(data_dir):
            if file_name.split('_')[0][-1].isdigit():
                continue

            market_file = data_dir.joinpath(file_name)
            file_hash = self._hash(file_path=market_file)

            if file_hash not in hash_mapping:
                raise ValueError(f'{file_name} does not match with any ticker in {list(hash_mapping.values())}!')

            linked_file = hash_mapping[file_hash]
            file_links['_'.join(file_name.split('_')[:-1])] = linked_file

        return file_links

    @classmethod
    def _hash(cls, file_path: str | pathlib.Path, hasher=None) -> str:
        hasher = hashlib.sha256() if hasher is None else hasher
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)

        file_hash = hasher.hexdigest()
        return file_hash

    def _prepare(self, market_file: str | pathlib.Path):
        market_log = pd.read_csv(market_file, encoding='GB2312')

        # Generate the timestamp
        market_log['timestamp'] = market_log.apply(lambda row: (datetime.datetime.strptime(str(row['交易日']) + ' ' + row['最后修改时间'], '%Y%m%d %H:%M:%S').timestamp() + row['最后修改毫秒'] / 1000), axis=1)

        # Filter out the unwanted columns and rename the remaining ones
        market_log = market_log.drop(columns=['交易日', '合约代码', '交易所代码', '合约在交易所的代码', '最后修改时间', '最后修改毫秒', '业务日期'])
        market_log = market_log.rename(columns=self.column_mapping)

        return market_log

    @PostgresClient.connect
    def _move_tables_to_schema(self, schema_name: str, table_prefix: str):
        conn = self.db_conn

        with conn.cursor() as cursor:
            cursor.execute(f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE '{table_prefix}%';""")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]
                cursor.execute(f'ALTER TABLE "public"."{table_name}" SET SCHEMA "{schema_name}";')
        conn.commit()

    def run(self):
        self.create_database()

        market_date = self.start_date
        logic_links = {}

        while market_date <= self.end_date:
            LOGGER.info(f"Processing data for {market_date}")

            data_dir = self.data_dir.joinpath(f'{market_date:%Y%m%d}')

            if not os.path.isdir(data_dir):
                market_date += datetime.timedelta(days=1)
                continue

            # step 1: rename files
            self._rename_files(data_dir)

            # step 2: upload the actual tickers
            for file_name in os.listdir(data_dir):
                if not file_name.split('_')[0][-1].isdigit():
                    LOGGER.info(f'{file_name} is skipped!')
                    continue

                market_file = data_dir.joinpath(file_name)
                market_log = self._prepare(market_file)
                table_name = file_name.replace('.csv', '')
                self.upload_df(data=market_log, table_name=table_name, index=False, method=self.psql_insert_copy)

            # step 3: generate file links
            file_links = self._hash_and_link_files(data_dir)
            logic_links[market_date] = {key.rstrip('_').upper(): value.split('_')[0] for key, value in file_links.items()}

            market_date += datetime.timedelta(days=1)

        # step -1: upload logical links
        self.upload_df(data=pd.DataFrame(logic_links).T, table_name='logical_links')

    def rearrange_tables(self):
        self.create_schema(schema_name='IC')
        self.create_schema(schema_name='IF')
        self.create_schema(schema_name='IH')
        self.create_schema(schema_name='IM')
        self.create_schema(schema_name='TF')
        self.create_schema(schema_name='TL')
        self.create_schema(schema_name='TS')
        self.create_schema(schema_name='T')

        self._move_tables_to_schema(schema_name='IC', table_prefix='IC')
        self._move_tables_to_schema(schema_name='IF', table_prefix='IF')
        self._move_tables_to_schema(schema_name='IH', table_prefix='IH')
        self._move_tables_to_schema(schema_name='IM', table_prefix='IM')
        self._move_tables_to_schema(schema_name='TF', table_prefix='TF')
        self._move_tables_to_schema(schema_name='TL', table_prefix='TL')
        self._move_tables_to_schema(schema_name='TS', table_prefix='TS')
        self._move_tables_to_schema(schema_name='T', table_prefix='T')


class FutureTickQuery(PostgresClient):
    def __init__(self, **kwargs):
        super().__init__(db_name='stock_future', **kwargs)

        self.logical_links = None

    @PostgresClient.connect
    def _query_ticker(self, logical_ticker: str, market_date: datetime.date) -> str:
        logical_symbol = logical_ticker.split('.')[0]

        if not self.logical_links:
            query = f'SELECT * FROM "public"."logical_links"'
            data = pd.read_sql(query, self.engine)
            data.set_index('index', inplace=True)

            self.logical_links = data.to_dict(orient='dict')

        return self.logical_links[logical_symbol][market_date]

    def __call__(self, ticker: str, market_date: datetime.date, dtype: str | type | None = None) -> list[TickData]:
        if dtype is not None and dtype != 'TickData':
            raise ValueError('Can only query TickData')

        ticker = ticker.upper()
        # the ticker is "IC2301.CFE"
        # Check if the ticker is in the format "IC_main.CFE" or similar
        if '_' in ticker and not ticker.split('_')[0][-1].isdigit():
            ticker = self._query_ticker(logical_ticker=ticker, market_date=market_date)

        # The ticker is in the format "IC2301.CFE" or similar
        symbol = ticker.split('.')[0]
        schema_name = symbol[:-4]
        table_name = f"{symbol}_{market_date:%Y%m%d}"

        # Query the data from the appropriate table
        query = f'SELECT * FROM "{schema_name}"."{table_name}"'
        data = pd.read_sql(query, self.engine)
        tick_data_list = []

        for _, log in data.iterrows():
            tick_data = TickData(
                ticker=symbol + '.CFE',
                timestamp=log['timestamp'],
                last_price=log['last_price'],
                bid_price=log['bid_price_1'],
                bid_volume=log['bid_volume_1'],
                ask_price=log['ask_price_1'],
                ask_volume=log['ask_volume_1'],
                pre_close=log['prev_close_price'],
                lower_limit_price=log['lower_limit_price'],
                upper_limit_price=log['upper_limit_price'],
                total_traded_volume=log['volume'],
                total_traded_notional=log['turnover']
            )
            tick_data_list.append(tick_data)

        return tick_data_list


class StockTransactionUploader(PostgresClient):
    column_mapping = {
        'symbol': 'ticker',
        'turnover': 'notional',
        'record_id': 'transaction_id',
        'order_kind': 'order_type',
        'ask_order_id': 'sell_id',
        'bid_order_id': 'buy_id'
    }

    def __init__(self, start_date: datetime, end_date: datetime.date, **kwargs):
        self.start_date = start_date
        self.end_date = end_date

        super().__init__(db_name='stock', **kwargs)
        assert os.name == 'posix', 'Can only run uploaded on the server. a Linux env is expected!'
        self.data_dir = pathlib.Path('/', 'mnt', 'FileServer', 'shared', 'xtp_data')

    def unzip(self, archive_path: str | pathlib.Path, output_dir: str | pathlib.Path):
        if not os.path.isfile(archive_path):
            raise ValueError(f'Archive {archive_path} not exist!')

        os.makedirs(output_dir, exist_ok=True)

        try:
            # Use pigz to decompress the file and tar to extract it
            subprocess.run(['tar', '--use-compress-program=pigz', '-xvf', archive_path, '-C', output_dir], check=True)
            LOGGER.info(f"Successfully extracted {archive_path} to {output_dir}")
        except subprocess.CalledProcessError as e:
            LOGGER.info(f"Error during extraction: {e}")

    def clean_up(self, market_date: datetime.date):
        unzip_dir = self.data_dir.joinpath(f'{market_date:%Y-%m-%d}')

        if os.path.isdir(unzip_dir):
            shutil.rmtree(unzip_dir)

    def _load_transaction(self, market_date: datetime.date) -> dict[str, pd.DataFrame]:
        tgz_file = self.data_dir.joinpath(f'{market_date:%Y-%m-%d}.tgz')
        output_dir = self.data_dir.joinpath(f'{market_date:%Y-%m-%d}')
        transaction_df = {}

        if not os.path.isfile(tgz_file):
            return transaction_df

        if os.path.isdir(output_dir):
            LOGGER.info(f'Output dir {output_dir} already exist! Unzip skipped!')
        else:
            self.unzip(archive_path=tgz_file, output_dir=self.data_dir)

        for file_name in os.listdir(transaction_dir := output_dir.joinpath('transactions')):
            if file_name.startswith('60'):
                market_log = self._prepare_transaction(file_name=transaction_dir.joinpath(file_name), market_prefix='60')
            elif file_name.startswith('30'):
                market_log = self._prepare_transaction(file_name=transaction_dir.joinpath(file_name), market_prefix='30')
            elif file_name.startswith('00'):
                market_log = self._prepare_transaction(file_name=transaction_dir.joinpath(file_name), market_prefix='00')
            else:
                continue

            if market_log.empty:
                transaction_df[file_name.rstrip('.csv')] = market_log

        return transaction_df

    def _prepare_transaction(self, file_name: str | pathlib.Path, market_prefix='') -> pd.DataFrame:
        original_log = pd.read_csv(file_name)

        if original_log.empty:
            original_log['timestamp'] = []
            original_log['side'] = []
        else:
            original_log['timestamp'] = original_log.apply(lambda row: (datetime.datetime.strptime(str(row['action_day']) + ' ' + str(row['time']), '%Y%m%d %H%M%S%f').timestamp()), axis=1)
            original_log['side'] = [1 if _ == 66 else -1 if _ == 83 else 0 for _ in original_log['bsflag']]

        # Filter out the unwanted columns and rename the remaining ones
        market_log = original_log.drop(columns=['action_day', 'time', 'order_kind', 'direction', 'datetime', 'bsflag', 'function_code', 'channel'])
        market_log = market_log.rename(columns=self.column_mapping)

        return market_log

    def run(self):
        self.create_database()

        market_date = self.start_date

        while market_date <= self.end_date:
            self.create_schema(schema_name=f'{market_date:%Y%m%d}')
            LOGGER.info(f"Processing data for {market_date}")
            tgz_file = self.data_dir.joinpath(f'{market_date:%Y-%m-%d}.tgz')
            output_dir = self.data_dir.joinpath(f'{market_date:%Y-%m-%d}')

            if not os.path.isfile(tgz_file):
                market_date += datetime.timedelta(days=1)
                continue

            if os.path.isdir(output_dir):
                LOGGER.info(f'Output dir {output_dir} already exist! Unzip skipped!')
            else:
                self.unzip(archive_path=tgz_file, output_dir=self.data_dir)

            for i, file_name in enumerate(ttl := os.listdir(transaction_dir := output_dir.joinpath('transactions'))):
                if file_name.startswith('60'):
                    market_log = self._prepare_transaction(file_name=transaction_dir.joinpath(file_name), market_prefix='60')
                elif file_name.startswith('30'):
                    market_log = self._prepare_transaction(file_name=transaction_dir.joinpath(file_name), market_prefix='30')
                elif file_name.startswith('00'):
                    market_log = self._prepare_transaction(file_name=transaction_dir.joinpath(file_name), market_prefix='00')
                else:
                    continue

                if market_log.empty:
                    continue

                ticker = file_name.rstrip('.csv')

                LOGGER.info(f'[{i} / {len(ttl)}] {market_date} {ticker} {len(market_log)} records to upload!')
                self.upload_df(data=market_log, table_name=f'{ticker}', schema=f'{market_date:%Y%m%d}', index=False, method=self.psql_insert_copy)

            self.clean_up(market_date=market_date)
            market_date += datetime.timedelta(days=1)


class StockTransactionQuery(PostgresClient):
    def __init__(self, **kwargs):
        super().__init__(db_name='stock', **kwargs)

    def __call__(self, ticker: str, market_date: datetime.date, dtype: str | type | None = None) -> list[TransactionData | TradeData]:
        if dtype is not None and dtype not in (expected_dtype := ('TransactionData', 'TradeData')):
            raise ValueError(f'{self.__class__} can only query {expected_dtype}!')

        # the ticker is "600010.SH"
        ticker = ticker.upper()
        transaction_data_list = []

        # The ticker is in the format "IC2301.CFE" or similar
        table_name = f"{ticker}"
        schema_name = f'{market_date:%Y%m%d}'

        # Query the data from the appropriate table
        try:
            query = f'SELECT * FROM "{schema_name}"."{table_name}"'
            data = pd.read_sql(query, self.engine)
        except sqlalchemy.exc.ProgrammingError as e:
            LOGGER.error(f'Query {ticker} {market_date} {dtype} failed!')

            if e.code != 'f405':
                LOGGER.error(f'\n{traceback.format_exc()}')

            return transaction_data_list

        for _, log in data.iterrows():
            if dtype == 'TradeData':
                if not int(log['side']):
                    continue

                constructor = TradeData
            elif dtype == 'TransactionData':
                constructor = TransactionData
            else:
                raise ValueError(f'Invalid dtype, expect TransactionData, got {dtype}!')

            data = constructor(
                ticker=ticker,
                timestamp=log['timestamp'],
                transaction_id=log['transaction_id'],
                buy_id=log['buy_id'],
                sell_id=log['sell_id'],
                side=int(log['side']),
                price=log['price'],
                volume=log['volume']
            )

            transaction_data_list.append(data)

        return transaction_data_list


def get_future_tick(ticker: str, market_date: datetime.date, dtype: str = None):
    query = FutureTickQuery()
    return query(ticker=ticker, market_date=market_date, dtype=dtype)


def main():
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 2, 1)

    # Initialize and run the uploader
    # uploader = FutureTickUploader(start_date, end_date)
    # uploader.run()
    # uploader.rearrange_tables()

    # query = FutureTickQuery()
    # query(market_date=datetime.date(2023, 2, 1), ticker='IC_main.cfe')

    # uploader = StockTransactionUploader(start_date, end_date)
    # uploader.run()

    # client = PostgresClient(db_name='stock')
    # client.delete_empty_schemas()

    # query = StockTransactionQuery()
    # r = query(market_date=datetime.date(2023, 2, 1), ticker='000002.SZ', dtype='TradeData')

    LOGGER.info('All done!')
    sys.exit(0)


if __name__ == '__main__':
    main()
