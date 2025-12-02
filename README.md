# Modbus Data Acquisition System

> **Note**: This repository is temporarily public for academic review as part of a Master's program application. All code is Copyright © 2025 Frank-Lucas Pantazis, All Rights Reserved. See [LICENSE](LICENSE) for details.

# Background
During one of my internships, I was tasked with developing a data-acquisition (DAQ) script for a Modbus device. 
The device was connected to a small single-board computer (similar to a Raspberry Pi) where it shared computing resources with three other sensors. 
The DAQ script was designed to run daily, with a `cron` job terminating it at **8:00 PM** and restarting it the following day at **5:00 AM**.

Design Constraints:
- Provide **1-second sampling** using a **lock-free implementation**
- Store **average values** of the signal to reduce disk usage.  
- Save averaged data **at least once per minute** to minimize data loss in case of a crash.  
- Record both **diagnostic** and **daily diagnostic** information.  
- Allow the sampling, averaging, and diagnostic intervals to be **user-defined**.
- Modbus TCP communication (simulated - original implementation used Modbus RTU over serial)

---

# DAQ Program
Multi-threaded DAQ for continuous sensor data collection with automated averaging.

## Overview
This system collects data from Modbus-based sensors.

- Samples sensor readings continuously
- Calculates fixed interval averages
- Logs diagnostic information periodically
- Writes data to timestamped CSV files every **1 minute**
- Handles interruptions and Shut-downs gracefully without data loss (completes writes and empties caches)

Two main threads orchestrate operation:

- Foreground thread: reads raw measurements and diagnostic data periodically
- Background thread: computes averages at the averaging interval and writes both averages and diagnostics periodically

The system is configurable via YAML and supports timezone customization. See `sensors_cfg/sensor.yaml` for more details.
Unit tests are built with pytest.

### Configurable Settings

- Timezone  
- Communication settings  
- Working directory and storage path  
- Time intervals (sampling, averaging, diagnostic)  
- Fields for daily diagnostic, diagnostic, and target data  
- Modbus data fields  

### Features
- CSV output:
  - Hourly subdirectories for averages/diagnostics data
  - Daily file for daily diagnostics (overwrite behavior within same day) 
- Robust average calculation:
  - Handles missing data between samples
  - Skips duplicate and out-of-order timestamps
  - Aligns average timestamp to end-interval convention
- Graceful shutdown that drains caches and completes writes

## Architecture

### System Design
```
┌─────────────────┐
│  Modbus Sensor  │
└────────┬────────┘
         │
┌────────▼────────────┐
│ Foreground Thread   │ ◄── sampling_interval
│ (read_continuously) │
└────────┬────────────┘
         │
         ▼
    ┌────────┐
    │ Caches │ ◄── Ring buffers
    └────┬───┘
         │
┌────────▼────────────┐
│ Background Thread   │ ◄── averaging_interval
│   (background)      │     WRITING_INTERVAL
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ CSV Files           │
└─────────────────────┘

```
## Project Structure
```
├── README.md
├── requirements.txt
├── daq.py 
├── main.py
├── sensor_simulator.py
├── utils.py
├── sensors_cfg/
│       └── sensor.yaml
└── tests/
      ├── avg_calc_test.py
      ├── data_operations_test.py
      └── test_sensor.yaml
```
### Key Files

**`daq.py`**  
Core data acquisition system. Manages two worker threads for continuous sensor reading and data processing. Handles timing, caching, averaging, and CSV output.

**`utils.py`**  
Defines `DataField`, `DataGroup`, and `DataSchema` classes for managing Modbus register layouts and converting between raw register values and typed Python data.

**`sensor_simulator.py`**  
Modbus TCP server that simulates a real sensor. Updates register values periodically to mimic live sensor readings.

**`main.py`**  
Application entry point. Starts the sensor simulator in a background thread, loads DAQ configuration from YAML, and begins data acquisition.

**`sensors_cfg/sensor.yaml`**  
Configuration file defining sensor parameters, Modbus connection settings, field definitions, and acquisition intervals.


## Output Structure
```
storage_path/
  ├── 202511/                # Year
  │      └── 11/              # Month
  │           └── 30/         # Day
  │                ├── 22/    # Hour
  │                │    ├── 00_diagnostics_2025113     # Diagnostic
  │                │    └── 20251130_22.csv            # Averaged Data
  │                └── 00_daily_diagnostics.csv        # Daily summary
  └── logs/
        └── sensor_daq_20251130_222513.log
```

## Requirements
- Python 3.10+
- Linux recommended (tested)
- Dependencies:
  - pymodbus
  - pytz
  - pyyaml
  - pytest

## Quick Start
Select an available TCP port and update the value in the configuration file.
Ensure the Modbus device (or simulator) is reachable from your system.

```bash
# Install dependencies
pip install -r requirements.txt

# Start the program
python main.py
```

Data will be written to the configured storage path organized by date/time.
Press Ctrl+C to stop gracefully to stop gracefully.

---

## Testing

- Test framework: pytest
- Run all tests:

```bash
pytest -q
```

- Coverage highlights:
  - Data conversion and schema (utils.py)
  - Average calculation with boundary, duplicate, out-of-order and float microsecond timestamps

Focus file examples:

- tests/avg_calc_test.py — averaging behaviors including float timestamps
- tests/data_operations_test.py — data conversion and grouping

---

## License

**All Rights Reserved**

Copyright © 2025 Frank-Lucas Pantazis

This project is provided solely for academic review as part of a Master's program 
application. This code is based on proprietary work and is **not available for use, 
modification, or distribution**.

Unauthorized use is prohibited. See [LICENSE](LICENSE) for details.
