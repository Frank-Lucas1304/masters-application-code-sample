import pytest
import os
from pymodbus.client import ModbusTcpClient
from daq import Daq, DECIMAL_PLACES
from datetime import datetime, timedelta
from pathlib import Path


@pytest.fixture()
def daq(mocker):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    mock_client = mocker.patch('daq.ModbusTcpClient')

    # Configure what the mock instance should return
    mock_instance = mock_client.return_value
    mock_instance.connect.return_value = True
    yaml_file = Path(__file__).parent / "test_sensor.yaml"
    daq = Daq.load(yaml_file)
    if os.path.exists("logs"): # keeps directory clean when testing
        os.rmdir("logs")
    # Required conditions for tests to be valid
    daq.sampling_interval = timedelta(seconds=1)
    daq.averaging_interval = timedelta(seconds=5)
    return daq


class TestAverageCalculation:
    def test_mock_connection(self,daq):
        assert daq.client.connect()

    def test_cache_assignment(self, daq):
        assert daq.raw_data_cache is not None

    def test_calculate_average_with_no_missing_values(self, daq):
        daq.cleaned_up = True
        tz_info = daq.timezone_info

        measured_data = [(1, 1),(2, 2),(3,3),(4,4),(5,5),(6,6)]
        values_within_window = [1,2,3,4,5]
        daq.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data, tz_info))

        daq.calculate_and_cache_average_value()
        expected_average = sum(values_within_window) / len(values_within_window)
        assert daq.average_cache[0][1] == round(expected_average, DECIMAL_PLACES)

    def test_calculate_average_with_float_timestamps(self, daq): #TODO fix
        daq.cleaned_up = True
        tz_info = daq.timezone_info

        measured_data = [(1, 2),(1.6, 3), (2.5, 5), (3.9, 4), (5.1, 6), (6, 4)]
        values_within_window = [2,3,5,4,6]
        daq.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data, tz_info))

        daq.calculate_and_cache_average_value()
        expected_average = sum(values_within_window) / len(values_within_window)
        assert daq.average_cache[0][1] == round(expected_average,DECIMAL_PLACES)

    def test_calculate_average_with_measurements_near_boundaries(self, daq):
        daq.cleaned_up = True
        tz_info = daq.timezone_info

        measured_data = [(4.6, 4),(1, 4)]
        values_within_window = [4]
        daq.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data, tz_info))

        daq.calculate_and_cache_average_value()
        expected_average = sum(values_within_window) / len(values_within_window)
        assert daq.average_cache[0][1] == round(expected_average, DECIMAL_PLACES)

    def test_calculate_average_when_raw_cache_has_multiple_values_at_same_second(self, daq):
        daq.cleaned_up = True
        tz_info = daq.timezone_info

        measured_data = [(1, 2), (1, 3), (2, 5), (2, 4), (2, 6), (3, 4)]
        values_within_window = [2,5,4]
        daq.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data, tz_info))

        daq.calculate_and_cache_average_value()

        expected_average = sum(values_within_window) / len(values_within_window)
        assert daq.average_cache[0][1] == round(expected_average,DECIMAL_PLACES), "Average was calculated with all values"
        # or
    def test_calculate_average_with_five_measurements_in_same_second(self,daq):
        daq.cleaned_up = True
        tz_info = daq.timezone_info

        measured_data = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
        values_within_window = [1]
        daq.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data, tz_info))

        daq.calculate_and_cache_average_value()

        expected_average = sum(values_within_window) / len(values_within_window)
        assert daq.average_cache[0][1] == round(expected_average,DECIMAL_PLACES), "Average was calculated with all values"

    def test_when_raw_cache_data_begins_at_time_zero(self,daq): # time zero not best description
        daq.cleaned_up = True

        tz_info = daq.timezone_info
        measured_data = [(0,2),(1,4),(2,5),(3,6),(4,6)]
        daq.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data, tz_info))
        daq.calculate_and_cache_average_value()
        assert len(daq.raw_data_cache) == 4, "Most values were not used since calculation was for previous window"
        assert daq.average_cache[0][1] == 2, "Only value at time 0 was used"

    def  test_no_average_calculated_when_only_single_measurement_exists(self,daq):
        daq.cleaned_up = True

        tz_info = daq.timezone_info
        ts = datetime(2020, 2, 1, 10, 10, 1, tzinfo=tz_info)
        reg_val = ModbusTcpClient.convert_to_registers(2, ModbusTcpClient.DATATYPE.FLOAT32)
        daq.raw_data_cache.append([ts,reg_val])

        daq.calculate_and_cache_average_value()
        assert len(daq.average_cache) == 0, "Should not calculate average with single value"
        assert len(daq.raw_data_cache) == 1, "Raw cache should still contain the single measurement"


    def test_when_raw_cache_size_2(self,daq):
        daq.cleaned_up = True
        # (0-5], parenthesis means not inclusive
        # Averaging happens in 5-second windows: (0-5], (0-5], etc.
        # sec=1 falls in (0-5] window, sec=6 falls in (5-10] window
        tz_info = daq.timezone_info
        measured_data = [(1,2),(6,4)] # format (sec, measured_value)
        daq.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data,tz_info))

        daq.calculate_and_cache_average_value()

        assert len(daq.raw_data_cache) == 1, "Raw cache should contain the sec=6 measurement (incomplete 5-10s window)"
        assert len(daq.average_cache) == 1, "Average cache should contain one value from completed (0-5]s window"
        assert daq.average_cache[0][1] == 2, "Average of (0-5]s window should be 2 (only one value)"

    def test_when_irr_cache_missing_data_at_start(self,daq): # time zero not best description
        daq.cleaned_up = True

        tz_info = daq.timezone_info
        mock_raw_data = [(2,3),(3,4),(4,5),(5,6),(6,7)]
        values_within_window = [3,4,5,6]

        daq.raw_data_cache.extend(TestAverageCalculation.create_samples(mock_raw_data,tz_info))
        daq.calculate_and_cache_average_value()

        expected_average = sum(values_within_window) / len(values_within_window)  # 4.6

        assert len(daq.raw_data_cache) == 1, "Should not contain measurement at 6s"
        assert daq.average_cache[0][1] == expected_average, "Average was calculated with values only within the (0-5]s window"


    def test_when_raw_cache_missing_data_in_middle(self,daq):
        daq.cleaned_up = True

        tz_info = daq.timezone_info
        measured_data = [(1,2),(3,3),(4,4),(5,6),(6,6)]
        values_within_window = [2,3,4,6]
        daq.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data,tz_info))

        expected_average = sum(values_within_window) / len(values_within_window)

        daq.calculate_and_cache_average_value()
        assert len(daq.raw_data_cache) == 1, "Should not contain measurement at 6s"
        assert daq.average_cache[0][1] == expected_average, "Average was calculated with values only within the (0-5]s window"

    def test_when_raw_cache_missing_data_at_end(self,daq):
        daq.cleaned_up = True

        tz_info = daq.timezone_info
        measured_data = [(1, 1),(1, 2), (3, 3), (6, 6), (7, 6)]
        values_within_window = [1, 2, 3]
        daq.raw_data_cache.extend(TestAverageCalculation.create_samples(measured_data,tz_info))
        expected_average = sum(values_within_window) / len(values_within_window)
        daq.calculate_and_cache_average_value()

        assert len(daq.average_cache) == 1
        assert daq.average_cache[0][1] == expected_average

    @staticmethod
    def create_samples(shorthand_data,tz_info):
        data =[]
        for sec, val in shorthand_data:
            ts = datetime(2020, 2, 1, 10, 10, int(sec),int(sec%1*10**6), tzinfo=tz_info)
            reg_val = ModbusTcpClient.convert_to_registers(val, ModbusTcpClient.DATATYPE.FLOAT32)
            data.append([ts, reg_val])
        return data
