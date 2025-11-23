from pymodbus.client import ModbusTcpClient
from datetime import tzinfo

class DataField:
    """
    Describes how to extract and convert data from a Modbus register block.
    """
    data_type_mapping_dict = {'F': 'FLOAT', 'S': 'INT', 'U': 'UINT'}  # from yaml to modbus
    register_length = 16

    def __init__(self, name: str, address: int, count: int, datatype: ModbusTcpClient.DATATYPE, header: list[str]=None):
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


    def convert(self, raw_register_list: list[int], address_offset: int=0,decimal_places:int=3):
        """
        Extracts the relevant raw registers and converts them to a typed value.

        :param decimal_places:
        :param raw_register_list: List[int]
            The full list of raw register values read from the Modbus device.
        :param address_offset: int, optional (default=0)
            An index offset to adjust the register addresses if the read block does not start at 0.

        :return: object, converted data.
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

    def revert(self,converted_value_list:list[int | float | str],idx: int=0):
        if idx >= len(converted_value_list):
            raise IndexError(f"Index {idx} out of bounds for list of length {len(converted_value_list)}")

        value = converted_value_list[idx]
        return ModbusTcpClient.convert_to_registers(value, data_type=self.datatype)

    def __apply_special_conversion(self, value: int | float | str):
        # TODO - put somewhere else (maybe)
        if 'date' in self.name:
            value = DataField.int_date_to_str(value)
        if self.name == 'dome_heating_status':
            value = 'On' if value else 'Off'
        elif self.name == 'alert_humidity' or self.name == 'alert_temperature':
            value = 'Normal' if value == 0 else 'Abnormal'
        return value


    @staticmethod
    def int_date_to_str(settings_date: int):
        # initial format is YYYYMMDD
        if settings_date == 0:
            return '-'
        else:
            day = settings_date % 100
            month = (settings_date // 100) % 100
            year = settings_date // 10000

            if not (1 <= month <= 12 and 1 <= day <= 31):
                raise f'Invalid date: {settings_date}'

            return f'{day}/{month}/{year}'



    def __repr__(self):
        return f"DataField({self.name}, addr: {self.address}, count: {self.count}, datatype: {self.datatype})"

class DataSchema:
    """
    Defines how all data fields of a device are laid out in Modbus memory.
    """

    def __init__(self, yaml_data_fields: dict[str,dict[str,int | str]]):
        self.map = DataSchema.create_field_objects(yaml_data_fields)

    @staticmethod
    def create_field_objects(yaml_data_fields:dict[str,dict[str,int | str]]):
        data_def_dict = dict()
        for name in yaml_data_fields.keys():
            data_def_dict[name] = DataSchema.__create_data_fields(name,yaml_data_fields[name])
        return data_def_dict

    def define_group(self,field_names:list, timezone_info:tzinfo=None, decimal_places:int=3):
        return DataGroup(field_names, self.map, timezone_info,decimal_places)

    @staticmethod
    def __create_data_fields(name:str, field_info: dict[str,int | str]):
        # TODO add raise erros
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

class DataGroup:
    """
    A subset of data fields from Modbus registers, grouped based on logical operations.

    This allows convenient conversion and access for related fields as a single unit.
    """

    def __init__(self, field_names:list[str], field_map:dict[str,DataField | list[DataField]],timezone:tzinfo=None,decimal_places:int=3):
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
            self.field_name_to_idx[name] = idx # TODO which to pick dict or just list??
            if isinstance(field,list):
                self.headers.append(field[0].header)
                for subfield in field:
                    self.raw_reg_size = max([self.raw_reg_size, subfield.last_memory_address+1])
            else:
                self.headers.append(field.header)
                self.raw_reg_size = max([self.raw_reg_size,field.last_memory_address+1])
    def convert_all(self,raw_register_list:list[int]):
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

    def revert_all(self,converted_values_list:list[int | float | str]):

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
    def write_to_register(registers,field,values):
        for idx,reg_val in enumerate(values):
            registers[field.address+idx] = reg_val

    def get(self,name):
        return self.fields[self.field_name_to_idx[name]]