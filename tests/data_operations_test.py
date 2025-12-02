import pytest
from daq import *


class TestDataField:
    @pytest.fixture()
    def data_field_obj(self):
        return DataField("raw_value",2,2, ModbusTcpClient.DATATYPE.FLOAT32,"Raw Value [units]")

    def test_convert_from_register(self,data_field_obj):
        # Gets input from pymodbus library method
        value = ModbusTcpClient.convert_to_registers(0.04355,ModbusTcpClient.DATATYPE.FLOAT32)
        # Testing projects custom implementation
        output = data_field_obj.convert(value, address_offset=2)
        assert output == 0.044, "Converts register encoded data to 32 bit float." # rounding is expected

    def test_convert_to_register(self,data_field_obj):
        # Gets input from pymodbus library method
        value = ModbusTcpClient.convert_from_registers([0,1],ModbusTcpClient.DATATYPE.FLOAT32) # value = 1.401298464324817e-45
        # Testing projects custom implementation
        output = data_field_obj.revert([value])
        assert output == [0,1], "Converts register encoded data to 32 bit float "# rounding is expected

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
        assert len(group.fields) == 2, "Data field group should only have 2 fields"

    def test_group_header_generation(self, sample_schema,timezone_info):
        """Test that headers are correctly extracted from fields."""
        group = sample_schema.define_group(['serial_number', 'firmware_version'], timezone_info)

        # First header should be timestamps
        assert 'timestamp' in group.headers[0].lower()
        # Subsequent headers from field definitions
        assert len(group.headers) == 3, "There must be 3 headers "
        assert group.headers[1] == "Serial Number" and group.headers[2] == "Firmware Version", "The Seriral Number header should appear before the Firmware Version header."
