import pytest
from daq import *


class TestDataField:
    # TODO - TEST HEADER GENERATION
    # TODO - Test count calculation
    # TODO - Add messafes after assert
    @pytest.fixture()
    def data_field_obj(self):
        return DataField("raw_value",2,2, ModbusTcpClient.DATATYPE.FLOAT32,"Raw Value [units]")

    def test_convert_from_register(self,data_field_obj):
        value = ModbusTcpClient.convert_to_registers(0.04355,ModbusTcpClient.DATATYPE.FLOAT32)
        output = data_field_obj.convert(value, address_offset=2)
        assert output == 0.044 # rounding is expected

    def test_convert_to_register(self,data_field_obj):
        value = ModbusTcpClient.convert_from_registers([0,1],ModbusTcpClient.DATATYPE.FLOAT32)
        output = data_field_obj.revert([value])
        assert output == [0,1]# rounding is expected

class TestDataGroup:
    @pytest.fixture()
    def sample_schema(self):
        with open("tests/test_sensor.yaml", "r") as f:
            config = yaml.safe_load(f)
        return DataSchema(config["registers"]["data_fields"])

    @pytest.fixture()
    def timezone_info(self):
        with open("tests/test_sensor.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config["timezone"]

    def test_group_field_count(self, sample_schema):
        """Test that field group contains correct number of fields."""
        group = sample_schema.define_group(['serial_number', 'firmware_version'])
        assert len(group.fields) == 2

    def test_group_header_generation(self, sample_schema,timezone_info):
        """Test that headers are correctly extracted from fields."""
        group = sample_schema.define_group(['serial_number', 'firmware_version'], timezone_info)

        # First header should be timestamps
        assert 'timestamp' in group.headers[0].lower()
        # Subsequent headers from field definitions
        assert len(group.headers) == 3  # timestamp + 2 fields

    def test_convert_all_returns_correct_count(self, sample_schema,timezone_info):

        """Test that convert_all returns one value per field."""
        group = sample_schema.define_group(['serial_number', 'firmware_version'],timezone_info)

        # Mock register data
        mock_registers = [0] * 20
        converted = group.convert_all(mock_registers)

        assert len(converted) == 2  # One per field