from pymodbus.client import ModbusTcpClient
from datetime import tzinfo
from typing import TypeVar

N = TypeVar("N", int, float)

class DataField:
    """
    Describes how to extract and convert data from a Modbus register block.
    """
    data_type_mapping_dict = {'F': 'FLOAT', 'S': 'INT', 'U': 'UINT'}  # from yaml to modbus
    register_length = 16

    def __init__(self, name: str, address: int, count: int, datatype: ModbusTcpClient.DATATYPE, header: str=None):
        """
        :param name: field description/name
        :param address: starting address in memory
        :param count: number of registers the information is stored in
        :param datatype: format the data should be converted to
        """

        if address < 0:
            raise ValueError(f"Address must be non-negative, got {address}")
        if count <= 0:
            raise ValueError(f"Count must be positive, got {count}")

        self.name = name
        self.address = address
        self.count = count
        self.datatype = datatype
        self.last_memory_address = address + count - 1
        self.header = header


    def convert(self, raw_register_list: list[int], address_offset: int=0,decimal_places:int=3) -> N | str:
        """
        Extracts the relevant raw registers and converts them to a typed value.

        :param raw_register_list: The full list of raw register values read from the Modbus device.
        :param address_offset: An index offset to adjust the register addresses if the read block does not start at 0. (default: 0)
        :param decimal_places: rounding precision (default: 3)

        :return: converted values.
        """

        start = self.address - address_offset
        end = self.last_memory_address - address_offset + 1

        if start < 0 or end > len(raw_register_list):
            raise IndexError(
                f"Register range [{start}:{end}] out of bounds for list of length {len(raw_register_list)}"
            )
        raw_slice = raw_register_list[start:end]

        if not isinstance(raw_slice, list):
            raw_slice = [raw_slice]

        value = ModbusTcpClient.convert_from_registers(raw_slice, data_type=self.datatype)
        value = self.__apply_special_conversion(value)

        if not isinstance(value, str):
            value = round(value, decimal_places)
        return value

    def revert(self,converted_value_list:list[N | str],idx: int=0) -> list[int]:
        """
        Converts data to raw 16 bit int register format

        :param converted_value_list: previously converted values
        :param idx: position in list of data element to revert

        :return: encoded 16-bit register values
        """
        if idx >= len(converted_value_list):
            raise IndexError(f"Index {idx} out of bounds for list of length {len(converted_value_list)}")

        value = converted_value_list[idx]
        return ModbusTcpClient.convert_to_registers(value, data_type=self.datatype)

    def __apply_special_conversion(self, value: N | str) -> N | str:
        """
        Apply model-specific conversions for certain fields.

        Special Conversion:
         - for dates: YYYYMMDD -> "YYYY/MM/DD"
         - for alerts:       0 -> Normal
                             1 -> Abnormal

        :param value: converted value
        :return: data with special conversion applied
        """

        if 'date' in self.name:
            value = DataField.int_date_to_str(value)
        elif self.name == 'alert_humidity' or self.name == 'alert_temperature':
            value = 'Normal' if value == 0 else 'Abnormal'
        return value


    @staticmethod
    def int_date_to_str(settings_date: int) -> str:
        """
        Converts int date to string.

        :param settings_date: date stored as settings in modbus: YYYYMMDD
        :return: string format of date "YYYY/MM/DD"
        """
        # initial format is YYYYMMDD
        if settings_date == 0:
            return '-'
        else:
            day = settings_date % 100
            month = (settings_date // 100) % 100
            year = settings_date // 10000

            if not (1 <= month <= 12 and 1 <= day <= 31):
                raise f'Invalid date: {settings_date}'

            return f'{year}/{month}/{day}'

    def __repr__(self):
        return f"DataField({self.name}, addr: {self.address}, count: {self.count}, datatype: {self.datatype})"

class DataGroup:
    """
    A subset of data fields from Modbus registers, grouped based on logical operations.

    This allows convenient conversion and access for related fields as a single unit.
    """

    def __init__(self, field_names:list[str], field_map:dict[str,DataField | list[DataField]],timezone:tzinfo=None,decimal_places:int=3):
        """
        Create a data group from a list of field names.

        :param field_names: ordered list of field names to include in the group.
        :param field_map: mapping of field names to DataField objects or lists for repeated fields.
        :param timezone: timezone used for timestamp labels in CSV output.
        :param decimal_places: rounding precision (default: 3).
        """
        self.fields = []
        self.field_name_to_idx = dict()
        self.raw_reg_size = -1
        self.decimal_places = decimal_places  # Store it here
        if timezone is None:
            self.headers = []
        else:
            self.headers = [f"timestamp [{timezone}]"]

        for idx,name in enumerate(field_names):
            field = field_map[name]
            self.fields.append(field)
            self.field_name_to_idx[name] = idx
            if isinstance(field,list):
                self.headers.append(field[0].header)
                for subfield in field:
                    self.raw_reg_size = max([self.raw_reg_size, subfield.last_memory_address+1])
            else:
                self.headers.append(field.header)
                self.raw_reg_size = max([self.raw_reg_size,field.last_memory_address+1])
    def convert_all(self,raw_register_list:list[int])-> list[N | str | list[N | str]]:
        """
       Convert all fields in this group from raw registers.

        :param raw_register_list: list of values that will be converted
        :return: list of values converted
        """
        converted_values = []
        for field in self.fields:
            # Purpose of following if condition is to display daily_diagnostics data in csv a certain way
            # All fields that have the repeat tag > 0 in the yaml file will enter this if statement
            if isinstance(field,list):
                sub_list = []
                group = field
                for reg in group:
                    sub_list.append(reg.convert(raw_register_list,decimal_places=self.decimal_places))
                converted_values.append(sub_list)
            else:
                converted_values.append(field.convert(raw_register_list,decimal_places=self.decimal_places))
        return converted_values

    def revert_all(self,converted_values_list:list[N | str]) -> list[N | str]:
        """
        Convert all group values back into a raw Modbus register block.

        :param converted_values_list: previously converted values

        :return: encoded 16-bit register values
        """
        raw_register_values = [-1 for _ in range(self.raw_reg_size)]
        for i, field in enumerate(self.fields):
            if isinstance(field,list):
                converted_subfield_list = converted_values_list[i]

                for j, subfield in enumerate(field):
                    reg_values = subfield.revert(converted_subfield_list,j)
                    DataGroup.write_to_register(raw_register_values,subfield,reg_values)
            else:
                reg_values = field.revert(converted_values_list,i)
                DataGroup.write_to_register(raw_register_values, field, reg_values)

        return raw_register_values
    @staticmethod
    def write_to_register(registers:list[int],field:DataField,values:list[int]) -> None:
        """
        Write a sequence of register values into the appropriate positions.

        :param registers: raw register array to assign values
        :param field: target field
        :param values: encoded 16-bit register values
        """

        for idx,reg_val in enumerate(values):
            registers[field.address+idx] = reg_val

    def get(self,name:str) ->  DataField | list[DataField]:
        """
        Retrieve the DataField (or list of DataField) corresponding to a field name.
        :param name: field name

        :return: DataField or list of DataField
        """
        return self.fields[self.field_name_to_idx[name]]



class DataSchema:
    """
    Defines the complete set of fields for a device based on YAML configuration.
    """

    def __init__(self, yaml_data_fields: dict[str,dict[str,int | str]]):
        """
        Load and construct all data fields from a YAML dictionary.

        :param yaml_data_fields: parsed YAML field definitions
        """
        self.map = DataSchema.create_field_objects(yaml_data_fields)

    @staticmethod
    def create_field_objects(yaml_data_fields:dict[str,dict[str,int | str]]) -> dict[str, DataField | list[DataField]]:
        """
        Create all DataFields from YAML definitions.

        :param yaml_data_fields: parsed YAML field definitions

        :return: Mapping of field names to DataField objects
        """
        data_def_dict = dict()
        for name in yaml_data_fields.keys():
            data_def_dict[name] = DataSchema.__create_data_fields(name,yaml_data_fields[name])
        return data_def_dict

    def define_group(self,field_names:list, timezone_info:tzinfo=None, decimal_places:int=3) -> DataGroup:
        """
        Create a DataGroup consisting of the specified field names.

        :param field_names: names of the fields to include
        :param timezone_info: timezone information for timestamp labels.
        :param decimal_places: rounding precision(default: 3)

        :return: DataGroup
        """
        return DataGroup(field_names, self.map, timezone_info,decimal_places)

    @staticmethod
    def __create_data_fields(name:str, field_info: dict[str,int | str]) -> DataField | list[DataField]:
        """
        Internal helper to construct a single field or repeated fields
        based on YAML settings.

        :param name: field name
        :param field_info: yaml field definition

        :return: Constructed field(s)
        """
        header = None
        data_format = field_info['format']  # required
        start_address = field_info['address']  # required
        if "header" in field_info.keys():
            header =  field_info['header']
        if data_format == 'STR':
            datatype = ModbusTcpClient.DATATYPE.STRING
            count = field_info['count']  # required if str
        else:
            datatype = DataField.data_type_mapping_dict[data_format[0]]
            data_size = int(data_format[1:])
            datatype = getattr(ModbusTcpClient.DATATYPE, f"{datatype}{data_size}")
            count = data_size // DataField.register_length

        # Repeat: repetition of the same type of field in Modbus (i.e. calibration date)
        # Gap: space between repeated data field (i.e. for calibration date, calibration value is in between)
        repeat = field_info.get('repeat',0)
        gap = field_info.get('gap',0)

        if repeat == 0:
            return DataField(name, start_address, count, datatype,header=header)

        data_objects = []
        for i in range(repeat):
            key = f'{name}{i}'
            address = start_address + (gap + count) * i
            data_objects.append(DataField(key, address, count, datatype,header=header))

        return data_objects