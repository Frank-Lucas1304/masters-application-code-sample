"""
Data Acquisition (DAQ) module for Modbus-based sensor systems.

Provides continuous data collection with configurable sampling, averaging, and
diagnostic intervals. Data is automatically written to CSV files organized by
timestamp.
"""

import atexit, signal
import collections
import csv, yaml
import os, sys
from pathlib import Path
from types import FrameType
import pytz
import time
from datetime import datetime, timedelta, date
from pymodbus.client import ModbusBaseSyncClient
import logging, threading
from utils import *

N = TypeVar("N", int, float)

detected_local_tz = datetime.now().astimezone().tzinfo

OBLIGATORY_CFG_KEYS = ['sensor_name', 'sensor_model', 'communication_settings','daq','registers']

DECIMAL_PLACES:int = 3
MAX_READ_REGISTER = 20
CACHE_SIZE_FACTOR = 5 # TODO Evaluate
WRITING_INTERVAL = timedelta(seconds=60) # TODO CHANGE

logger = logging.getLogger("daq")


class Daq:
    """
    Data Acquisition system for Modbus-based sensors with multi-threaded operation.

    Manages continuous sensor data collection with configurable sampling, averaging,
    and diagnostic intervals. Uses two worker threads for concurrent operation:
    - Foreground: Reads sensor at sampling_interval
    - Background: Computes averages and writes data to CSV files

    All data is written to timestamped CSV files organized by month/day/hour.
    Graceful shutdown ensures no data loss on interruption.
    """

    def __init__(self, modbus_client_obj:ModbusBaseSyncClient,storage_path:str,fields_cfg: dict[str, dict[str, str | int]], field_groups:dict[str,list[str]],
                 sampling_interval=timedelta(seconds=1),
                 averaging_interval=timedelta(seconds=10), diagnostic_interval=timedelta(seconds=60),
                 timezone_info=detected_local_tz):
        """
        Initializes Data Acquisition Instance for Modbus sensor

        :param modbus_client_obj: modbus client object
        :param storage_path: path to which data series will be stored; folders for month, day, hour will be created
        :param fields_cfg: modbus fields definition
        :param field_groups: defines list of fields in each group: target, diagnostics and daily_diagnostics
        :param sampling_interval: datetime timedelta -- temporal window between raw data series
        :param averaging_interval: datetime timedelta -- temporal window for average data series
        :param diagnostic_interval: datetime timedelta -- temporal window for diagnostic data series
        :param timezone_info: Specify pytz.timezone of the data and log file. If not provided computer timezone is used.
        """

        # MODBUS COMMUNICATION PROTOCOL
        self.client = modbus_client_obj

        # DAQ SETTINGS
        self.storage_path = storage_path
        self.diagnostic_filename = f'{datetime.now().astimezone(timezone_info):00_diagnostics_%Y%m%d_%H%M%S}'
        self.sampling_interval = sampling_interval
        self.averaging_interval = averaging_interval
        self.diagnostic_interval = diagnostic_interval
        self.__record_status = self.validating_params()

        # AVG TARGET SIGNAL
        self.raw_data_cache = self.init_cache()
        self.average_cache = []
        self.diagnostic_cache = []

        # DATA FIELD INFO
        self.fields = DataSchema(fields_cfg)
        self.target_fields = self.fields.define_group(field_groups['target'],timezone_info,DECIMAL_PLACES)
        self.diagnostic_fields = self.fields.define_group(field_groups['diagnostics'],timezone_info,DECIMAL_PLACES)
        self.daily_diagnostic_fields = self.fields.define_group(field_groups['daily_diagnostics'],decimal_places=DECIMAL_PLACES)

        # TIME REFERENCE
        # Assume program is restarted every day no need to update value
        self.timezone_info = timezone_info
        self.midnight = datetime.combine(date.today(), datetime.min.time()).astimezone(timezone_info)

        # CLEANUP + THREAD KILL
        self.stop_event = threading.Event()
        self.stop_event.clear()

        self.cleaned_up = False # Ensure cleanup is only run once (signal and atexit have overlapping triggers)
        curr_os = sys.platform
        self.write_lock = threading.Lock()
        if curr_os == 'win32':
            signal.signal(signal.SIGBREAK, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        signal.signal(signal.SIGINT, self.cleanup)
        atexit.register(self.cleanup)


    def validating_params(self) -> bool:
        """
        Validates that the DAQ time parameters meet program requirements.
        These requirements were implemented to simply its design.

        Requirements:
        - Averaging interval must be >= sampling interval
        - Diagnostic interval must be >= sampling interval
        - Averaging interval must be an exact multiple of sampling interval
        - Diagnostic interval must be an exact multiple of averaging interval

        :return: True if all parameters are valid, False otherwise.
        """
        size_constraint = (self.averaging_interval.seconds >= self.sampling_interval.seconds and self.diagnostic_interval.seconds >= self.sampling_interval.seconds)
        # Necessary constraint due to program design
        factor_constraint = self.averaging_interval.seconds % self.sampling_interval.seconds == 0 and self.diagnostic_interval.seconds % self.averaging_interval.seconds == 0

        if not size_constraint:
            logger.error("The averaging_interval and the diagnostic_interval must be greater or equal to the sampling time.")
        if not factor_constraint:
            logger.error('The sampling time must be a factor of the averaging_interval and the diagnostic interval')
        return size_constraint and factor_constraint

    def init_cache(self) -> collections.deque:
        """
        Creates a ring buffer for storing raw measurement data.

        The buffer size is calculated based on the writing interval and sampling
        interval to ensure sufficient capacity between disk writes.

        :return: Ring buffer with max size
        """
        if not self.__record_status:
            raise ValueError('self.sampling.seconds need to be a factor of self.averaging_interval.seconds')
        max_size = int(CACHE_SIZE_FACTOR * WRITING_INTERVAL.seconds // self.sampling_interval.seconds)
        return collections.deque(maxlen=max_size)

    def write_daily_diagnostics(self) -> None:
        """
        Writes daily diagnostics data to disk.

        """
        ts = datetime.now().astimezone(self.timezone_info)
        data = self.get_daily_diagnostics()
        self.write_data_to_csv(ts, data, '00_daily_diagnostics', self.daily_diagnostic_fields.headers,daily_diagnostics=True)

    def get_corrected_measurement(self) -> list[int]:
        """
        Reads target measurement from sensor.

        :return: A list containing raw 16-bit register values representing the measured data.
        """
        data_def = self.fields.map['corr_value']
        return self.client.read_holding_registers(data_def.address, count=data_def.count).registers

    def get_daily_diagnostics(self) -> list[ N | str | list[list[str | float]]]:
        """
        Reads daily diagnostic registers and reshapes specific fields (calibration history
        and linear coefficients) into a CSV-friendly structure with nested lists.

        Notes:
            - Current CSV-structure is specifically for model X sensors.
            - Adjustment might be necessary for future versions of program if different models are added to project

        :return: CSV-formatted list where each element represents a field:
                    - Scalar values (int, float, str) for simple fields
                    - Nested lists for calibration history: [[date, value], ...]
                    - Nested lists for coefficients: [['k0', val], ['k1', val], ...]
        """
        calibration_history, lin_coefficients, csv_formated_data = [] , [], []
        count = self.daily_diagnostic_fields.raw_reg_size
        reg = self.client.read_holding_registers(0, count=count)
        register_list = reg.registers

        converted_data = self.daily_diagnostic_fields.convert_all(register_list)
        field_name_to_idx = self.daily_diagnostic_fields.field_name_to_idx

        for idx in range(len(converted_data)):
            value = converted_data[idx]
            if field_name_to_idx['lin_coefficient_k'] == idx:
                for i, coefficient in enumerate(value):
                    lin_coefficients.append([f'k{i}', coefficient])
                converted_data[idx] = lin_coefficients
                csv_formated_data.append(lin_coefficients)

            elif field_name_to_idx['calibration_date_k'] == idx:
                calibration_values = converted_data[field_name_to_idx["calibration_value_k"]]
                for i, calibration_date in enumerate(value):
                    calibration_value = calibration_values[i]
                    if calibration_date == '-':
                        calibration_value = '-'
                    calibration_history.append([calibration_date,calibration_value])
                converted_data[idx] = calibration_history
                csv_formated_data.append(calibration_history)

            elif field_name_to_idx['calibration_value_k'] == idx:
                # Skipped since calibration values are collected with the calibration dates in the elif calibration_date_k bloc.
                # Doing so ensure that the formatting for this csv file is independent of the order in which fields appear in the converted_data list.
                continue
            else:
                csv_formated_data.append(converted_data[idx])

        return csv_formated_data

    def get_measurement_and_diagnostic_data(self) -> tuple[list[int],list[int]]:
        """
        Reads raw measurement and diagnostic data from sensor.

        returns: A tuple containing:
                - measurements: List of 16-bit register values for sensor readings
                - diagnostics: List of 16-bit register values for diagnostic data
        """
        count = self.daily_diagnostic_fields.raw_reg_size - 1
        reg = self.client.read_holding_registers(0, count=count)
        data_def = self.fields.map['corr_value']
        measurement = reg.registers[data_def.address:data_def.count]
        diagnostics = reg.registers
        return measurement, diagnostics

    def get_and_cache_measurement(self, timestamp:datetime) -> None:
        """
        Reads raw measurement and adds it to cache.

        :param timestamp: time at which the values were read from sensor.
        """
        now = datetime.now().astimezone(self.timezone_info)
        logger.info(f'measurement - retrieved data at {now}')
        irr = self.get_corrected_measurement()
        logger.debug(f'measurement - data taken before {datetime.now().astimezone(self.timezone_info)}')
        self.raw_data_cache.append([timestamp, irr])
        logger.debug(f'measurement - stored data before {datetime.now().astimezone(self.timezone_info)}')

    def get_and_cache_measurement_and_diagnostic(self, timestamp:datetime) -> None:
        """
        Reads raw measurement and diagnostic data. Then appends the values to their respective cache.

        :param timestamp: time at which the values were acquired.
        """
        now = datetime.now().astimezone(self.timezone_info)
        logger.info(f'measurement + diagnostics - retrieved data at {now}')
        measurement, diagnostic_data = self.get_measurement_and_diagnostic_data()
        logger.debug(f'measurement + diagnostics - data taken before {datetime.now().astimezone(self.timezone_info)}')
        self.raw_data_cache.append([timestamp, measurement])
        self.diagnostic_cache.append([timestamp, diagnostic_data])
        logger.debug(f'measurement + diagnostics - stored data before {datetime.now().astimezone(self.timezone_info)}')

    def convert_diagnostic_cache_data(self) -> list[list[str | N ]]:
        """
        Converts raw register data in diagnostic cache.

        :return: Converted diagnostic data.
        """
        convert_rows = []
        while len(self.diagnostic_cache):
            ts, raw_data = self.diagnostic_cache.pop(0)
            timestamp = ts.strftime('%Y-%m-%d %H:%M:%S.%f')

            converted_data = self.diagnostic_fields.convert_all(raw_data)
            row = [timestamp]
            row.extend(converted_data)
            convert_rows.append(row)
        return convert_rows

    def calculate_and_cache_average_value(self, cleanup:bool =False) -> None:
        """
        Calculates the average of measurements within the current averaging interval.

        Processes raw measurements from the cache and computes the mean value aligned
        to averaging interval boundaries. Handles missing data by adjusting for
        actual time gaps between samples.

        :param cleanup: if True, processes all remaining cached data (used during shutdown)
        """
        # TODO - CHECK HOW IT RESPONDS TO FLOATS NEAR BOUNDARIES
        if len(self.raw_data_cache) <= 1 and not cleanup:  # check to avoid any data racing
            logger.debug('Calculating average measurement is not possible at the moment')
            return

        i = 0
        avg = 0
        ts, _ = self.raw_data_cache[0]
        prev_ts = ts
        start_ts = ts  # For debugging purposes
        end_ts = ts  # For debugging purposes
        wait_time, _ = self.calculate_interval_timing(self.averaging_interval, ts)
        seconds_remaining_in_averaging_interval = round(wait_time, 0)
        first_loop = True
        # ---- Handles Starting --- #
        if seconds_remaining_in_averaging_interval == self.averaging_interval.seconds:  # this means that the starting data point is at 0
            seconds_remaining_in_averaging_interval = 0

        avg_measurement_ts = ts + timedelta(
            seconds=seconds_remaining_in_averaging_interval)  # Ensures that the timestamp for the average is independent of the data. (Useful if missing values)

        while len(self.raw_data_cache):
            ts, raw_measurement = self.raw_data_cache[0]
            measurement = self.target_fields.get('corr_value').convert(raw_measurement)
            # This section determines if the average should be computed now or later. It takes into consideration missing data.
            #   - i.e: Let the sampling time be 1s. If a timestamps is skipped then time_gap will adjust and have a value of 2s.
            time_gap = ts - prev_ts
            # ---- Handles Duplicate Timestamps ---- #
            if ts == prev_ts and not first_loop or ts < prev_ts: # time_gap might be slightly off from zero due to substraction operation
                self.raw_data_cache.popleft()
                continue
            first_loop = False

            # ---- Handles Missing Data (uneven timegaps) ----- #
            seconds_remaining_in_averaging_interval -= round(time_gap.total_seconds(), 0)
            prev_ts = ts

            # ---- Handles Average Calculation ----- #
            if seconds_remaining_in_averaging_interval >= 0:
                avg += measurement
                i += 1
                self.raw_data_cache.popleft()
                end_ts = ts
            else:
                if i==0:
                    avg = measurement
                    i = 1
                    self.raw_data_cache.popleft()
                break

        self.average_cache.append([avg_measurement_ts.strftime('%Y-%m-%d %H:%M:%S.%f'), round(avg / i, DECIMAL_PLACES)])
        logger.info(
            f'Calculated average measurement @ {avg_measurement_ts:%Y-%m-%d %H:%M:%S} - using {i} points collected during [{start_ts:%Y-%m-%d %H:%M:%S} to {end_ts:%Y-%m-%d %H:%M:%S}]')

    def write_data_to_csv(self, ts:datetime, data:list[str | N | list[str | N ]], filename_format:str, headers:list[str], clear:bool=True, daily_diagnostics:bool=False) -> None:
        """
        Writes data to disk in csv file format.

        :param ts: current time
        :param data: data to be stored in csv file.
        :param filename_format: time based formating string
        :param headers: aliases for the column names.
        :param clear: if True, frees space. (default is True) #TODO Clarify
        :param daily_diagnostics: if True, formats data into specific template for daily diagnostics. (default is False)
        """

        target_folder = f'{self.storage_path}/{ts:%Y}{ts:%m}/{ts:%d}/{ts:%H}/'
        if daily_diagnostics:
            target_folder = f'{self.storage_path}/{ts:%Y}{ts:%m}/{ts:%d}/'

        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)

        target_file = f'{target_folder}/{ts.strftime(filename_format)}.csv'
        file_mode = 'w' if daily_diagnostics else 'a'  # Ensures that the daily diagnostics file gets replaced if the script was run more than once in a day

        if os.path.exists(target_file) and not daily_diagnostics:
            headers = None
        with self.write_lock:
            with open(target_file, file_mode, newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',escapechar="\\")

                if daily_diagnostics:  # handles csv setup
                    for idx, row in enumerate(data):
                        header = headers[idx]
                        if isinstance(row, list):
                            writer.writerow([header])
                            writer.writerows(row)
                        else:
                            writer.writerow([header, row])
                else:
                    if headers:
                        writer.writerow(headers)
                    writer.writerows(data)
                logger.info(
                    f"data written in {ts.strftime(filename_format)}.csv before {datetime.now().astimezone(self.timezone_info)}")

            if clear:  # Clear cache if needed
                data.clear()

    def read_continuously(self) -> None:
        """
        Reads continuously raw data from sensor and appends it to the respective cache.

        Notes
        -----
        The validation constraint requiring diagnostic_interval to be an exact multiple of
        sampling_interval ensures diagnostics only trigger at times when samples are actually
        taken (i.e., at their LCM, which equals diagnostic_interval when this constraint holds).
        """
        while not self.stop_event.is_set():
            wait_time, time_since_midnight = self.calculate_interval_timing(self.sampling_interval)
            if self.stop_event.wait(timeout=wait_time):
                break
            try:
                ts = datetime.now().astimezone(self.timezone_info)
                # Calculate target time we intended to hit (before any execution delay)
                target_time_seconds = int(time_since_midnight.total_seconds() + wait_time)
                is_diagnostics = (target_time_seconds % self.diagnostic_interval.seconds) == 0
                if is_diagnostics:
                    self.get_and_cache_measurement_and_diagnostic(ts)
                else:
                    self.get_and_cache_measurement(ts)

            except Exception as e:
                logger.error(f"Foreground thread crashed: {e}",exc_info=True)
                self.stop_event.set()


    def background(self) -> None:
        """
        Calculates average signal and writes it and diagnostic data to csv files.

        """

        time.sleep(
            self.averaging_interval.seconds)  # Additional wait is to ensure that the measurement queue is always populated before computing the average
        while not self.stop_event.is_set():
            try:
                ts = datetime.now().astimezone(self.timezone_info)
                write_wait_time, write_tsm = self.calculate_interval_timing(WRITING_INTERVAL,ts)
                calc_wait_time, calc_tsm = self.calculate_interval_timing(self.averaging_interval,ts)
                wait_time = min([write_wait_time, calc_wait_time])
                if self.stop_event.wait(timeout=wait_time): # also could exit early if ever flag is raised
                    # stop_event was set! Exit immediately
                    break

                target_time_seconds = int(calc_tsm.total_seconds() + wait_time)
                is_calc_time = (target_time_seconds % self.averaging_interval.seconds) == 0

                if is_calc_time:
                    self.calculate_and_cache_average_value()

                target_time_seconds = int(write_tsm.total_seconds() + wait_time)
                is_write_time = (target_time_seconds % WRITING_INTERVAL.seconds) == 0

                if is_write_time:
                    if len(self.average_cache) > 0:
                        ts_str, _ = self.average_cache[0]
                        ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
                        self.write_data_to_csv(ts, self.average_cache, "%Y%m%d_%H", self.target_fields.headers)
                    else:
                        logger.warning("no data in average cache to write")

                    if len(self.diagnostic_cache) > 0:
                        ts, *_ = self.diagnostic_cache[0]
                        converted_diagnostic_data = self.convert_diagnostic_cache_data()
                        self.write_data_to_csv(ts, converted_diagnostic_data, '00_diagnostics_%Y%m%d_%H',
                                               self.diagnostic_fields.headers)
                    else:
                        logger.warning("no data in diagnostic cache to write")

            except Exception as e:
                logger.error(f"Background thread crashed: {e}",exc_info=True)
                self.stop_event.set()


    def start_acquisition(self) -> None:
        """
        Starts the acquisition process.

        """
        print("Initializing acquisition process...")

        if not self.__record_status:
            logger.info(
                'status - conditions to record image not met. Change frequency of sampling, averaging and/or diagnostics.')
            return

        if not self.client.connect():  # Checks if a device is connected to serial port
            logger.info(
                'status - conditions to record data not met. Unable to establish connection with pyranometer.')
            return

        self.write_daily_diagnostics()

        # self.midnight = datetime.combine(date.today(), datetime.min.time()).astimezone(
        #     self.timezone_info)  # Since program is restarted every day no need to update value
        wait_time, _ = self.calculate_interval_timing(self.averaging_interval)
        logger.info(f"Daq program will begin in {round(wait_time,3)} seconds")

        time.sleep(wait_time)
        # Ensure that the following code starts at timestamp where seconds_into_averaging_interval = 0
        try:
            foreground_thread = threading.Thread(target=self.read_continuously,daemon=True)
            background_thread = threading.Thread(target=self.background,daemon=True)

            foreground_thread.start()
            background_thread.start()
            foreground_thread.join()
            background_thread.join()

        except KeyboardInterrupt:
            logger.info("Daq stopped by user")
        except Exception as e:
            logger.error("Acquisition crashed", exc_info=True)
        finally:
            self.stop_event.set()


    def cleanup(self, sig:int | None =None, frame:FrameType | None =None) -> None:
        # assumed this method will only be executed before the end of the day (might be issue or else with file in which data is written too

        if self.cleaned_up:
            return
        self.cleaned_up = True
        logger.info('Cleaning up')
        if sig:
            logger.info(f"Initiated by signal: {sig}")
        else:
            logger.info(f"Initiated by atexit")
        self.stop_event.set()

        if self.write_lock.acquire(timeout=5):
            logger.info('File operations completed')
            self.write_lock.release()
        else:
            logger.warning('File operations not completed')

        logger.debug('Calculating average measurement with remaining data in cache')
        logger.debug(f'Before Cleanup - measurement cache\'s size:{len(self.raw_data_cache)}')
        samples_per_average = self.averaging_interval.seconds // self.sampling_interval.seconds
        logger.debug(
            f"Before Cleanup - Average cache\'s size: {len(self.average_cache)}, at least {len(self.raw_data_cache) / samples_per_average} are to be added ")

        while len(self.raw_data_cache):
            self.calculate_and_cache_average_value(cleanup=True)

        logger.debug(f"Before Cleanup - Average cache\'s size: {len(self.average_cache)}")

        if len(self.average_cache) == 0:
            return
        ts_str, _ = self.average_cache[0]
        ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
        self.write_data_to_csv(ts, self.average_cache, "%Y%m%d_%H",self.target_fields.headers )
        self.client.close()

    def calculate_interval_timing(self, reference_interval:timedelta, ts:datetime=None) -> tuple[float,timedelta]:
        """
        Calculates the remaining time left in the interval and the elapsed time since midnight from the given timestamp.

        :param reference_interval: the reference time interval.
        :param ts: the timestamp used to perform the calculations. If `None`, the current time is used.
        :return: A tuple where:
            - The first element is the remaining time in the current interval.
            - The second element is the elapsed time since midnight.
        """

        curr_ts = datetime.now().astimezone(self.timezone_info)
        if ts is None:
            ts = curr_ts
        time_since_midnight = ts - self.midnight
        # Must use total_seconds method to be able to use sleep since total_seconds includes sub-seconds info.
        seconds_into_averaging_interval = time_since_midnight.total_seconds() % reference_interval.seconds
        wait_time = reference_interval.seconds - seconds_into_averaging_interval
        return wait_time, time_since_midnight

    @staticmethod
    def load(yaml_file):
        storage_path = Path()
        daq_working_dir = Path()
        config=dict()
        with open(yaml_file, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)

        sensor_cfg = dict()
        comm_cfg = dict()
        fields_cfg = dict()
        field_groups = dict()
        daq_cfg = dict()
        for k in OBLIGATORY_CFG_KEYS:

            temp = config.get(k)
            if k == 'daq':
                daq_working_dir = Path(temp['daq_working_dir'])
                storage_path = temp['storage_path']
                if 'process' in temp.keys():
                    daq_cfg = temp['process']
                if 'groups' in temp.keys():
                    field_groups = temp['groups']
            elif k == 'communication_settings':
                comm_cfg['port'] = temp['port']
                comm_cfg['host'] = temp['host']
            elif k == 'registers':
                fields_cfg = temp['data_fields']
            else:
                sensor_cfg[k] = temp

        if type(daq_cfg) is dict:  # Converting to timedelta objects
            if 'sampling_interval' in daq_cfg.keys() and daq_cfg['sampling_interval'] is not None:
                daq_cfg['sampling_interval'] = timedelta(seconds=daq_cfg['sampling_interval'])

            if 'averaging_interval' in daq_cfg.keys() and daq_cfg['averaging_interval'] is not None:
                daq_cfg['averaging_interval'] = timedelta(seconds=daq_cfg['averaging_interval'])

            if 'diagnostic_interval' in daq_cfg.keys() and daq_cfg['diagnostic_interval'] is not None:
                daq_cfg['diagnostic_interval'] = timedelta(seconds=daq_cfg['diagnostic_interval'])

        if not os.path.isdir(daq_working_dir):
            os.makedirs(daq_working_dir)
        os.chdir(daq_working_dir)
        if not os.path.isdir("logs"):
            os.makedirs("logs")

        timezone_info = Daq.timezone_adjustment(config.get('timezone'))
        if timezone_info is None:
            timezone_info = detected_local_tz
        else:
            timezone_info = pytz.timezone(timezone_info)

        # return  sensor_cfg, fields_cfg, comm_cfg, daq_cfg , timezone_info
        timestamp_logfile = datetime.now(timezone_info)

        logging.basicConfig(filename=f'logs/daq_{timestamp_logfile:%Y%m%d_%H%M%S}.log',
                            level=logging.DEBUG, format='%(levelname)s - %(asctime)s: %(message)s', datefmt='%H:%M:%S')

        # STDOUT LOGGER
        root = logger
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        root.addHandler(console_handler)

        modbus_obj = ModbusTcpClient(**comm_cfg)
        return Daq(modbus_obj,storage_path,fields_cfg=fields_cfg,field_groups=field_groups,**daq_cfg,timezone_info=timezone_info)

    @staticmethod
    def timezone_adjustment(timezone_str:str) -> str:
        """
        Converts timezone string to pytz-compatible format.

        Handles GMT offset notation by reversing the sign (GMT+5 -> GMT-5)
        to match pytz's Etc/GMT convention.

        :param timezone_str: Timezone string (e.g., "GMT+5", "America/New_York")

        :return: Pytz-compatible timezone string
        """
        if "GMT" in timezone_str:
            if "+" in timezone_str:
                timezone_str = timezone_str.replace("+", "-")
            elif "-" in timezone_str:
                timezone_str = timezone_str.replace("-", "+")
            timezone_str = "Etc/"+ timezone_str
        return timezone_str



