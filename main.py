import time
from daq import Daq
import sensor_simulator as sim
import threading


if __name__ == "__main__":
    sensor_thread = None
    try:
        sensor_thread = threading.Thread(target=sim.sensor, daemon=True) # simulating sensor
        sensor_thread.start()
        time.sleep(5) # Wait time to ensure sensor is completely set up before beginning daq
        daq = Daq.load("sensors_cfg/sensor.yaml")
        daq.start_acquisition() # starting daq

    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error running server: {e}")
