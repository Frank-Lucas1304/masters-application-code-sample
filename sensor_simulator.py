import asyncio
import random
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusDeviceContext, ModbusServerContext
from pymodbus.server import ModbusTcpServer
from utils import *
import yaml

async def update_registers(holding_registers:ModbusSequentialDataBlock) -> None:
    """
    Periodically updates register values every 110ms to simulate sensor reading data.

    :param holding_registers: data object where raw values will be stored
    """
    while True:
        # Simulate new sensor readings
        temperature_values = [random.randint(4, 15) for _ in range(3)]
        reg_temp_values = ModbusTcpClient.convert_to_registers(temperature_values,ModbusTcpClient.DATATYPE.FLOAT32)

        holding_registers.setValues(1, reg_temp_values)

        await asyncio.sleep(0.110)


async def modbus_server(sensor_host:str,sensor_port:int) -> None:
    """
    Initializes modbus sever and updates its registers
    """

    # Define Modbus registers (initial data is empty)
    config = dict()
    with open("sensors_cfg/sensor.yaml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    modbus_fields = config.get("registers")
    fields = DataSchema(modbus_fields['data_fields'])

    field_values = [0,0,0,"X   \\",7,9,0,20240201,25348340,"Arbitrary_Sensor",[0,1,0,0],[20240201,0],[10,20]]

    registers = fields.define_group(list(fields.map.keys()))
    init_register = registers.revert_all(field_values)

    if len(init_register)%2==0:
        init_register = [0]+init_register

    holding_registers = ModbusSequentialDataBlock(0, init_register)  # Start with empty holding registers

    # Modbus slave context
    slave_context = ModbusDeviceContext(
        hr=holding_registers,
    )

    # Server context (holds the Modbus datastore)
    server_context = ModbusServerContext(devices=slave_context, single=True)

    # Start the Modbus TCP server
    server = ModbusTcpServer(
        context=server_context,
        address=(sensor_host, sensor_port),
    )

    # Start the background task to update registers periodically
    asyncio.create_task(update_registers(holding_registers))
    print(f"Connecting Virtual Sensor: Starting Modbus TCP Server on port {sensor_port}...")
    await server.serve_forever()  # This will block until the server is stopped

def sensor(sensor_host:str,sensor_port:int) -> None:
    """
    Runs sensor simulation
    """
    try:
        asyncio.run(modbus_server(sensor_host,sensor_port))  # Start the server in a managed event loop
    except Exception as e:
        print(f"Error running server: {e}")
    except KeyboardInterrupt:
        pass
