import threading
import time
from datetime import timezone

import pytest
from daq import *


#https://www.geeksforgeeks.org/python/unit-testing-python-unittest/
#https://www.geeksforgeeks.org/software-testing/how-to-do-unit-testing-detailed-guide-with-best-practices/
#https://www.geeksforgeeks.org/software-testing/types-software-testing/

# TODO SIMPLE DESIGN
#  - Ensure irradiance queue is never empty:
#       - have time offset between foreground and background thread
#  - long running code issues:
#       - kill and restart threads
#  - stress test thread (look up)


@pytest.fixture()
def sensor_receiver(mocker): #TODO CHANGE STORAGE PATH SUCH THAT IT IS POSSIBLE TO RUN TESTS
    mock_client = mocker.patch('daq.ModbusTcpClient')

    # Configure what the mock instance should return
    mock_instance = mock_client.return_value
    mock_instance.connect.return_value = True
    yaml_file = Path(__file__).parent / "test_sensor.yaml"
    sensor_receiver_obj = SensorReceiver.load(yaml_file)
    # Required conditions for tests to be valid
    sensor_receiver_obj.sampling_interval = timedelta(seconds=1)
    sensor_receiver_obj.averaging_interval = timedelta(seconds=5)
    return sensor_receiver_obj



class TestAverageCalculation:
    def test_mock_connection(self,sensor_receiver):
        assert sensor_receiver.client.connect()

    def test_cache_assignment(self, sensor_receiver):
        assert sensor_receiver.raw_data_cache is not None

    # ASSUMPTIONS:
    # - All values in the irr_cache have to be in total part of atleast 2 different average sampling intervals
    # - Each value can only be part of one averaging interval (FACT)
    # - There should always be one value left in irr_cache after average calculation (FACT)
    # - Might be able to counter this by figuring out the offset between background and foreground thread from the start? (However everything is controlled so the gap will always be the same)
    # TODO - FIX ISSUE
    # - Decimal Settings need to be shared
    # def test_calculate_average_when_raw_cache_has_multiple_values_at_same_second(self,sensor_receiver):
    #     tz_info = sensor_receiver.timezone_info
    #     measured_data = [(0,2),(0,2),(1,5),(2,6),(3,4),(4,4)]
    #     values_within_window = [2,5,6,4,4]
    #     sensor_receiver.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data, tz_info))
    #
    #     sensor_receiver.calculate_and_cache_average_value()
    #     expected_average = sum(values_within_window) / len(values_within_window)
    #     assert sensor_receiver.average_cache[0][1] == round(expected_average,DECIMAL_PLACES), "Average was calculated with all values"

    def test_calculate_average_when_raw_cache_has_multiple_values_at_same_second(self, sensor_receiver):
        tz_info = sensor_receiver.timezone_info
        measured_data = [(0, 2), (0, 3), (1, 5), (1, 4), (1, 6), (4, 4)]
        values_within_window = [2, 5, 4]
        sensor_receiver.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data, tz_info))

        sensor_receiver.calculate_and_cache_average_value()
        expected_average = sum(values_within_window) / len(values_within_window)
        assert sensor_receiver.average_cache[0][1] == round(expected_average,
                                                            DECIMAL_PLACES), "Average was calculated with all values"
        # or
    def test_calculate_average_with_five_measurements_in_same_second(self,sensor_receiver):
        pass

    def test_when_raw_cache_data_begins_at_time_zero(self,sensor_receiver): # time zero not best description
        tz_info = sensor_receiver.timezone_info
        measured_data = [(0,2),(1,4),(2,5),(3,6),(4,6)]
        values_within_window = [2,4,5,6,6]
        sensor_receiver.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data, tz_info))

        sensor_receiver.calculate_and_cache_average_value()
        expected_average = sum(values_within_window) / len(values_within_window)  # 4.6
        assert sensor_receiver.average_cache[0][1] == expected_average, "Average was calculated with all values"

    def  test_no_average_calculated_when_only_single_measurement_exists(self,sensor_receiver):
        tz_info = sensor_receiver.timezone_info
        ts = datetime(2020, 2, 1, 10, 10, 1, tzinfo=tz_info)
        reg_val = ModbusTcpClient.convert_to_registers(2, ModbusTcpClient.DATATYPE.FLOAT32)
        sensor_receiver.raw_data_cache.append([ts,reg_val])

        sensor_receiver.calculate_and_cache_average_value()
        assert len(sensor_receiver.average_cache) == 0, "Should not calculate average with single value"
        assert len(sensor_receiver.raw_data_cache) == 1, "Raw cache should still contain the single measurement"


    def test_when_raw_cache_size_2(self,sensor_receiver):
        # Averaging happens in 5-second windows: [0-5), [5-10), etc.
        # sec=1 falls in [0-5) window, sec=6 falls in [5-10) window
        tz_info = sensor_receiver.timezone_info
        measured_data = [(1,2),(6,4)] # format (sec, measured_value)
        sensor_receiver.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data,tz_info))

        sensor_receiver.calculate_and_cache_average_value()

        assert len(sensor_receiver.raw_data_cache) == 1, "Raw cache should contain the sec=6 measurement (incomplete 5-10s window)"
        assert len(sensor_receiver.average_cache) == 1, "Average cache should contain one value from completed 0-5s window"
        assert sensor_receiver.average_cache[0][1] == 2, "Average of 0-5s window should be 2 (only one value)"

    def test_when_irr_cache_missing_data_at_start(self,sensor_receiver): # time zero not best description
        tz_info = sensor_receiver.timezone_info
        mock_raw_data = [(2,3),(3,4),(4,5),(5,6),(6,7)]
        values_within_window = [3,4,5]

        sensor_receiver.raw_data_cache.extend(TestAverageCalculation.create_samples(mock_raw_data,tz_info))
        sensor_receiver.calculate_and_cache_average_value()

        expected_average = sum(values_within_window) / len(values_within_window)  # 4.6

        assert len(sensor_receiver.raw_data_cache) == 2, "Should not contain measurement at 5s or 6s"
        assert sensor_receiver.average_cache[0][1] == expected_average, "Average was calculated with values only within the 0-5s window"


    def test_when_raw_cache_missing_data_in_middle(self,sensor_receiver):
        tz_info = sensor_receiver.timezone_info
        measured_data = [(1,2),(3,3),(4,4),(5,6),(6,6)]
        values_within_window = [2,3,4]
        sensor_receiver.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data,tz_info))

        expected_average = sum(values_within_window) / len(values_within_window)

        sensor_receiver.calculate_and_cache_average_value()
        assert len(sensor_receiver.raw_data_cache) == 2, "Should not contain measurement at 5s or 6s"
        assert sensor_receiver.average_cache[0][1] == expected_average, "Average was calculated with values only within the 0-5s window"

    def test_when_irr_cache_missing_data_at_end(self,sensor_receiver):
        tz_info = sensor_receiver.timezone_info
        measured_data = [(1, 1),(1, 2), (3, 3), (5, 6), (6, 6)]
        values_within_window = [1, 2, 3]
        sensor_receiver.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data,tz_info))
        expected_average = sum(values_within_window) / len(values_within_window)
        sensor_receiver.calculate_and_cache_average_value()
        assert len(sensor_receiver.average_cache) == 1
        assert sensor_receiver.average_cache[0][1] == expected_average
    @staticmethod
    def create_samples(shorthand_data,tz_info):
        data =[]
        for sec, val in shorthand_data:
            ts = datetime(2020, 2, 1, 10, 10, sec, tzinfo=tz_info)
            reg_val = ModbusTcpClient.convert_to_registers(val, ModbusTcpClient.DATATYPE.FLOAT32)
            data.append([ts, reg_val])
        return data
