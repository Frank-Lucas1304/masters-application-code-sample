import csv
import os
import sys
import time
import logging
from datetime import datetime, timedelta, date
import yaml
import threading
from pathlib import Path
import traceback
import collections
import signal
import atexit
from utils import *
from pymodbus.client import ModbusBaseSyncClient,ModbusTcpClient
import pytz

detected_local_tz = datetime.now().astimezone().tzinfo

OBLIGATORY_CFG_KEYS = ['pyranometer_name', 'pyranometer_model', 'communication_settings', 'latitude', 'longitude',
                       'altitude', 'daq','groups','registers']

WRITING_INTERVAL = timedelta(seconds=10) # TODO CHANGE
CACHE_SIZE_FACTOR = 5 # TODO Evaluate
DECIMAL_PLACES = 3
MAX_READ_REGISTER = 20

# TODO
#  - 8  : Check Docstrings
#  - 10 : Finish Tests
#  - 9  : Read Me + add recording sample
#  - 1  : Check the Cache size factor and writing interval
#  - 4  : Lock Error when interrupting program ending thread --> try to handle interrupts
#  - 0  : Add Debug Table
#  - 8  : cleanup utils.py
#  - 1  : sign and add name or license to repo
#  - 3  : Maybe should add logg with Modbus debug and one wihtout


class SensorReceiver:

    def __init__(self, modbus_client_obj:ModbusBaseSyncClient, sensor_model:str,storage_path:str,register_map_setting:int,fields_cfg: dict[str, dict[str, str | int]], field_groups:dict[str,list[str]],
                 sampling_interval=timedelta(seconds=1),
                 averaging_interval=timedelta(seconds=10), diagnostic_interval=timedelta(seconds=60),
                 timezone_info=detected_local_tz):
        """
        Initializes SensorReceiver with user parameters

        :param storage_path: path to which data series will be stored; folders for month, day, hour will be created
        :param sampling_interval: datetime timedelta -- temporal offset between raw data series
        :param averaging_interval: datetime timedelta -- temporal window for average data series
        :param timezone_info: Specify pytz.timezone of the data and log file. If not provided computer timezone is used.
        """

        self.sensor_model = sensor_model

        # MODBUS COMMUNICATION PROTOCOL
        self.client = modbus_client_obj
        # DAQ
        self.storage_path = storage_path
        self.diagnostic_filename = f'{datetime.now():00_diagnostics_%Y%m%d_%H%M%S}'
        self.timezone_info = timezone_info
        self.midnight = None
        self.sampling_interval = sampling_interval
        self.averaging_interval = averaging_interval
        self.diagnostic_interval = diagnostic_interval
        self.__record_status = self.validating_params()


        # AVG IRR SIGNAL
        self.raw_data_cache = self.init_cache()
        self.average_cache = []
        self.diagnostic_cache = []

        # Kill Threads
        self.stop_event = threading.Event()
        self.stop_event.clear()

        # DATA FILED + REGISTER INFO
        self.register_map_setting = register_map_setting
        self.fields = DataSchema(fields_cfg)
        self.target_fields = self.fields.define_group(field_groups['target'],timezone_info,DECIMAL_PLACES)
        self.diagnostic_fields = self.fields.define_group(field_groups['diagnostics'],timezone_info,DECIMAL_PLACES)
        self.daily_diagnostic_fields = self.fields.define_group(field_groups['daily_diagnostics'],decimal_places=DECIMAL_PLACES)

        self.midnight = datetime.combine(date.today(), datetime.min.time()).astimezone(
            self.timezone_info)  # Since program is restarted every day no need to update value

        # CLEANUP
        curr_os = sys.platform
        self.write_lock = threading.Lock()
        if curr_os == 'win32':
            signal.signal(signal.SIGBREAK, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup) # type of signal the crontab script will send
        signal.signal(signal.SIGINT, self.cleanup)  # keyboard interrupt
        atexit.register(self.cleanup)


    def validating_params(self) -> bool:
        """
        Verifies that user chose appropriate time intervals. Sampling Synchronization (Not sure if this was necessary but doing so makes the data manipulation and retrieval easier to implement)

        :return: Validity of parameters.
        """
        size_constraint = (self.averaging_interval.seconds >= self.sampling_interval.seconds and self.diagnostic_interval.seconds >= self.sampling_interval.seconds)
        # Necessary constraint due to program design
        factor_constraint = self.averaging_interval.seconds % self.sampling_interval.seconds == 0 and self.diagnostic_interval.seconds % self.averaging_interval.seconds == 0 and WRITING_INTERVAL.seconds % self.averaging_interval.seconds == 0

        if not size_constraint:
            logging.error("The averaging_interval and the diagnostic_interval must be greater or equal to the sampling time.")
        if not factor_constraint:
            logging.error('The sampling time must be a factor of the averaging_interval and the diagnostic interval')
        return size_constraint and factor_constraint

    def init_cache(self):
        """
        Initializes a cache with a size limit according to design constraints. Specifically used for averaging calculations.

        :return: Queue with size limit.
        """
        if not self.__record_status:
            raise ValueError('self.sampling.seconds need to be a factor of self.averaging_interval.seconds')
        max_size = int(CACHE_SIZE_FACTOR * WRITING_INTERVAL.seconds // self.sampling_interval.seconds)
        return collections.deque(maxlen=max_size)

    def write_daily_diagnostics(self):
        ts = datetime.now().astimezone(self.timezone_info)
        data = self.get_daily_diagnostics()
        self.write_data_to_csv(ts, data, '00_daily_diagnostics', self.daily_diagnostic_fields.headers,daily_diagnostics=True)

    def get_corrected_measurement(self):
        """
        Gets measurement from sensor.

        """
        data_def = self.fields.map['corr_value']
        return self.client.read_holding_registers(data_def.address, count=data_def.count).registers

    def get_daily_diagnostics(self):
        count = self.daily_diagnostic_fields.raw_reg_size  # TODO weird ERROR expect multiple of 2
        reg = self.client.read_holding_registers(0, count=count)
        register_list = reg.registers
        print(register_list)

        converted_data = self.daily_diagnostic_fields.convert_all(register_list)
        # CSV FORMATING SPECIFIC FOR EKO MS-80SH
        # FROM IDX GET NAME -->
        calibration_history = []
        lin_coefficients = []
        field_name_to_idx = self.daily_diagnostic_fields.field_name_to_idx
        for idx in range(len(converted_data)):
            value = converted_data[idx]
            if field_name_to_idx['lin_coefficient_k'] == idx:
                for i, coefficient in enumerate(value):
                    lin_coefficients.append([f'k{i}', coefficient])
                converted_data[idx] = lin_coefficients

            if field_name_to_idx['calibration_date_k'] == idx:
                for i, calibration_date in enumerate(value):
                    calibration_history.append([calibration_date])
                converted_data[idx] = calibration_history

            if field_name_to_idx['calibration_value_k'] == idx:
                for i, calibration_value in enumerate(value):
                    if calibration_history[i][0] == '-':
                        calibration_value = '-'
                    calibration_history[i].append(calibration_value)

        return converted_data[:-1]

    def get_measurement_and_diagnostic_data(self):
        count = self.daily_diagnostic_fields.raw_reg_size - 1
        reg = self.client.read_holding_registers(0, count=count)
        data_def = self.fields.map['corr_value']
        measurement = reg.registers[data_def.address:data_def.count]
        diagnostics = reg.registers
        return measurement, diagnostics

    def get_and_cache_measurement(self, timestamp):
        now = datetime.now().astimezone(self.timezone_info)
        logging.info(f'measurement - retrieved data at {now}')
        irr = self.get_corrected_measurement()
        logging.info(f'measurement - data taken before {datetime.now().astimezone(self.timezone_info)}')
        self.raw_data_cache.append([timestamp, irr])
        logging.info(f'measurement - stored data before {datetime.now().astimezone(self.timezone_info)}')

    def get_and_cache_measurement_and_diagnostic(self, timestamp):
        now = datetime.now().astimezone(self.timezone_info)
        logging.info(f'measurement + diagnostics - retrieved data at {now}')
        measurement, diagnostic_data = self.get_measurement_and_diagnostic_data()
        logging.info(f'measurement + diagnostics - data taken before {datetime.now().astimezone(self.timezone_info)}')
        self.raw_data_cache.append([timestamp, measurement])
        self.diagnostic_cache.append([timestamp, diagnostic_data])
        logging.info(f'measurement + diagnostics - stored data before {datetime.now().astimezone(self.timezone_info)}')

    def convert_diagnostic_cache_data(self):
        convert_rows = []
        while len(self.diagnostic_cache):
            ts, raw_data = self.diagnostic_cache.pop(0)
            timestamp = ts.strftime('%Y-%m-%d %H:%M:%S.%f')

            converted_data = self.diagnostic_fields.convert_all(raw_data)
            row = [timestamp]
            row.extend(converted_data)
            convert_rows.append(row)
        return convert_rows

    def calculate_and_cache_average_value(self, cleanup=False):

        if len(self.raw_data_cache) <= 1 and not cleanup:  # check to avoid any data racing
            logging.debug('Calculating average measurement is not possible at the moment') # TODO WHAT HAPPENS IN THIS CASE
            return

        i = 0
        avg = 0
        ts, _ = self.raw_data_cache[0]
        prev_ts = ts
        start_ts = ts  # For debugging purposes
        end_ts = ts  # For debugging purposes
        wait_time, _ = self.calculate_interval_timing(self.averaging_interval, curr_ts=ts)
        seconds_remaining_in_averaging_interval = round(wait_time, 0)
        first_loop = True

        if seconds_remaining_in_averaging_interval == self.averaging_interval:  # this means that the starting data point is at 0
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
            if ts == prev_ts and not first_loop: # time_gap might be slightly off from zero due to substraction operation
                self.raw_data_cache.popleft()
                continue
            first_loop = False

            # ---- Handles Missing Data (uneven timegaps) ----- #
            seconds_remaining_in_averaging_interval -= round(time_gap.total_seconds(), 0)
            prev_ts = ts

            # ---- Handles Average Calculation ----- #
            if seconds_remaining_in_averaging_interval > 0: #TODO Important error removed - >= would have included the last value (needs to be >)

                avg += measurement
                i += 1
                self.raw_data_cache.popleft()
                end_ts = ts
            else:
                break

        self.average_cache.append([avg_measurement_ts.strftime('%Y-%m-%d %H:%M:%S.%f'), round(avg / i, DECIMAL_PLACES)])
        logging.info(
            f'Calculated average measurement @ {avg_measurement_ts:%Y-%m-%d %H:%M:%S} - using {i} points collected during [{start_ts:%Y-%m-%d %H:%M:%S} to {end_ts:%Y-%m-%d %H:%M:%S}]')

    def write_data_to_csv(self, ts, data, filename_format, headers, clear=True, daily_diagnostics=False):

        target_folder = f'{self.storage_path}/{ts:%m}/{ts:%d}/{ts:%H}/'
        if daily_diagnostics:
            target_folder = f'{self.storage_path}/{ts:%m}/{ts:%d}/'

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
                logging.debug(
                    f"data written in {ts.strftime(filename_format)}.csv before {datetime.now().astimezone(self.timezone_info)}")

            if clear:  # Clear cache if needed
                data.clear()

    def read_continuously(self):
        while not self.stop_event.is_set():
            wait_time, time_since_midnight = self.calculate_interval_timing(self.sampling_interval)
            if self.stop_event.wait(timeout=wait_time):
                break
            # time.sleep(wait_time)
            try:
                ts = datetime.now().astimezone(self.timezone_info)
                diagnostic_counter = int(
                    (time_since_midnight.total_seconds() + wait_time) % self.diagnostic_interval.seconds)
                if diagnostic_counter == 0:
                    self.get_and_cache_measurement_and_diagnostic(ts)
                else:
                    self.get_and_cache_measurement(ts)

            except Exception as e:
                logging.error(f"foreground_thread crashed: {e}")
                traceback.print_exc()
                self.stop_event.set()


    def background(self):

        measurement_write_counter = 0  # could use counters here because it does not have to be precise. (only the interval duration matters)
        diagnostic_write_counter = 0
        # FACT: sel.diagnostic_interval is a multiple of self.averaging_interval (this constraint was defined in self.validating_params()
        averaging_instances_per_writing_interval = WRITING_INTERVAL.seconds // self.averaging_interval.seconds
        averaging_instances_per_diagnostic_interval = averaging_instances_per_writing_interval
        if self.diagnostic_interval.seconds > WRITING_INTERVAL.seconds:
            averaging_instances_per_diagnostic_interval = self.diagnostic_interval.seconds // self.averaging_interval.seconds

        time.sleep(
            self.averaging_interval.seconds)  # Additional wait is to ensure that the measurement queue is populated when computing the average
        while not self.stop_event.is_set():
            try:
                wait_time, _ = self.calculate_interval_timing(self.averaging_interval)

                if self.stop_event.wait(timeout=wait_time): # also could exit early if ever flag is raised
                    # stop_event was set! Exit immediately
                    break
                # time.sleep(wait_time)  # makes sure that there is always an extra data point available

                self.calculate_and_cache_average_value()

                measurement_write_counter = (measurement_write_counter + 1) % averaging_instances_per_writing_interval
                diagnostic_write_counter = (diagnostic_write_counter + 1) % averaging_instances_per_diagnostic_interval

                if measurement_write_counter == 0:
                    ts_str, _ = self.average_cache[0]
                    ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
                    self.write_data_to_csv(ts, self.average_cache, "%Y%m%d_%H", self.target_fields.headers)

                if diagnostic_write_counter == 0:
                    if len(self.diagnostic_cache) > 0:
                        ts, *_ = self.diagnostic_cache[0]
                        converted_diagnostic_data = self.convert_diagnostic_cache_data()
                        self.write_data_to_csv(ts, converted_diagnostic_data, '00_diagnostics_%Y%m%d_%H',
                                               self.diagnostic_fields.headers)
                    else:
                        logging.warning("no diagnostic cache to write")
            except Exception as e:
                traceback.print_exc()
                logging.error(f"background_thread crashed: {e}")
                self.stop_event.set()


    def start_acquisition(self):

        if not self.__record_status:
            logging.info(
                'status - conditions to record image not met. Change frequency of sampling, averaging and/or diagnostics.')
            return

        if not self.client.connect():  # Checks if a device is connected to serial port
            logging.info(
                'status - conditions to record data not met. Unable to establish connection with pyranometer.')
            return

        self.write_daily_diagnostics()

        # self.midnight = datetime.combine(date.today(), datetime.min.time()).astimezone(
        #     self.timezone_info)  # Since program is restarted every day no need to update value
        wait_time, _ = self.calculate_interval_timing(self.averaging_interval)
        logging.info(f"Program will begin in {round(wait_time,3)} seconds")

        time.sleep(wait_time)
        # Ensure that the following code starts at timestamp where seconds_into_averaging_interval = 0
        try:
            foreground_thread = threading.Thread(target=self.read_continuously,daemon=True)
            background_thread = threading.Thread(target=self.background,daemon=True)

            foreground_thread.start()
            background_thread.start()
            foreground_thread.join()
        except KeyboardInterrupt:
            # Should I add anything
            pass
        except Exception as e:
            # Should I add anything
            pass
        finally:
            self.stop_event.set()


    def cleanup(self, sig=None, frame=None):
        # assumed that the this will be executed before the end of the day (might be issue or else with file in which data is written too
        logging.info('Cleaning up')
        self.stop_event.set()

        if self.write_lock.acquire(timeout=5):
            logging.info('File operations completed')
            self.write_lock.release()
        else:
            logging.warning('File operations not completed')

        logging.debug('Calculating average measurement with remaining data in cache')
        logging.debug(f'Before Cleanup - measurement cache\'s size:{len(self.raw_data_cache)}')
        samples_per_average = self.averaging_interval.seconds // self.sampling_interval.seconds
        logging.debug(
            f"Before Cleanup - Average cache\'s size: {len(self.average_cache)}, at least {len(self.raw_data_cache) / samples_per_average} are to be added ")

        while len(self.raw_data_cache):
            self.calculate_and_cache_average_value(cleanup=True)

        logging.debug(f"Before Cleanup - Average cache\'s size: {len(self.average_cache)}")

        if len(self.average_cache) == 0:
            return
        ts_str, _ = self.average_cache[0]
        ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
        self.write_data_to_csv(ts, self.average_cache, "%Y%m%d_%H",self.target_fields.headers )
        self.client.close()

    def calculate_interval_timing(self, reference_interval, curr_ts=None):
        """
        Calculates the remaining time left in the interval and the elapsed time since midnight.
        """
        ts = datetime.now().astimezone(self.timezone_info)
        if curr_ts is not None:
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

        sensor_cfg = {'location': {}}
        comm_cfg = dict()
        fields_cfg = dict()
        field_groups = dict()
        daq_cfg = dict()
        for k in OBLIGATORY_CFG_KEYS:

            temp = config.get(k)
            if k == 'daq':
                sensor_cfg['modbus_addr'] = temp['modbus_addr']
                storage_path = temp['storage_path']
                daq_working_dir = temp['daq_working_dir']
                if 'process' in temp.keys():
                    daq_cfg = temp['process']
                if 'groups' in temp.keys():
                    field_groups = temp['groups']
            elif k in ['latitude', 'longitude', 'altitude']:
                sensor_cfg['location'][k[:3]] = temp
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

        if not os.path.isdir('logs'):
            os.mkdir('logs')

        timezone_info = pytz.timezone("ETC/"+config.get('timezone')) # TODO THIS IS A PATCH
        if timezone_info is None:
            timezone_info = detected_local_tz

        timestamp_logfile = datetime.now()

        logging.basicConfig(filename=f'logs/pyranometer_daq_{timestamp_logfile:%Y%m%d_%H%M%S}.log',
                            level=logging.INFO, format='%(levelname)s - %(asctime)s: %(message)s', datefmt='%H:%M:%S')

        register_map_setting = config.get("register_map_setting")

        modbus_obj = ModbusTcpClient(**comm_cfg)
        return SensorReceiver(modbus_obj,"Sensor_X",storage_path,register_map_setting,fields_cfg=fields_cfg,field_groups=field_groups,**daq_cfg,timezone_info=timezone_info)


def main():
    sensor_obj = SensorReceiver.load("sensors/sensor.yaml")
    sensor_obj.start_acquisition()



if __name__ == '__main__':
    main()
