import asyncio
import random
import logging
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusDeviceContext, ModbusServerContext
from pymodbus.server import ModbusTcpServer
from utils import *
import yaml
logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)
from datetime import datetime


# Function to periodically update the registers
async def update_registers(holding_registers):
    while True:
        # Simulate new sensor readings (temperature values)
        temperature_values = [random.randint(4, 15) for _ in range(3)]
        reg_temp_values = ModbusTcpClient.convert_to_registers(temperature_values,ModbusTcpClient.DATATYPE.FLOAT32)
        # Update holding registers with new temperature values
        # TODO weird thing going on. When I index address 0 on the client side it gets address 1 here. I think the address 0 for the data block might represent something special
        #   -- Too look into
        holding_registers.setValues(1, reg_temp_values)
        #print(f"Updated temperature values: {temperature_values}")
        print(holding_registers.values)
        # Sleep for 5 seconds before updating again
        await asyncio.sleep(0.110)


async def run_server():
    # Define Modbus registers (initial data is empty)
    config = dict()
    with open("sensors/sensor.yaml","r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    modbus_fields = config.get("registers")
    fields = DataSchema(modbus_fields['data_fields'])
    # should be data and
    field_values = [0,0,0,"X   \\",7,9,0,20240201,25348340,"Arbitrary_Sensor",[0,1,0,0],[20240201,0],[10,20]]

    registers = fields.define_group(list(fields.map.keys()))
    print(registers.get("sensor_name"))
    init_register = registers.revert_all(field_values)

    # TODO Fix Register Issue (FIND MAYBE BETTER WAY)
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
        address=("0.0.0.0", 5020),  # Listen on all network interfaces
    )

    # Start the background task to update registers periodically
    asyncio.create_task(update_registers(holding_registers))

    # Start the additional task (e.g., logging)
    #asyncio.create_task(daq.main())

    print("Starting Modbus TCP Server on port 5020...")
    await server.serve_forever()  # This will block until the server is stopped


if __name__ == "__main__":
    # Ensure we're running the event loop
    try:
        asyncio.run(run_server())  # Start the server in a managed event loop
    except Exception as e:
        print(f"Error running server: {e}")
