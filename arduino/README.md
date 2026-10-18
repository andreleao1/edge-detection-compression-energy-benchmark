# Arduino + INA219 — Energy Monitoring with Prometheus & Grafana

Monitors the power consumption of a Raspberry Pi (or any DC load) using an **Arduino UNO** and an **INA219** current/power sensor. Metrics are exposed as a Prometheus endpoint and visualised in Grafana.

## Hardware Wiring

The INA219 communicates with the Arduino via I2C. The shunt resistor is placed **in series** with the load so the sensor can measure current flow.

The DS18B20 measures ambient temperature alongside each energy sample. Connect its data wire to Arduino digital pin **4**, and use a **4.7 kΩ pull-up resistor** between data and 5V. Connect VDD to 5V and GND to GND. Install the Arduino libraries **OneWire** and **DallasTemperature** in addition to **Adafruit INA219**.

![INA219 wiring diagram](assets/ina_diagram.jpg)

| INA219 Pin | Arduino Pin |
|---|---|
| VCC | 5V |
| GND | GND |
| SCL | A5 (SCL) |
| SDA | A4 (SDA) |
| Vin+ | Load supply + |
| Vin− | Load (Raspberry Pi) + |

> The shunt resistor sits between **Vin−** and the positive terminal of the load. The power supply negative goes directly to the load negative.

## Prerequisites

- Docker and Docker Compose
- Arduino UNO with INA219 module wired as above
- DS18B20 wired to digital pin 4 with a 4.7 kΩ data-line pull-up
- Arduino IDE (to flash `ina219_without_ethernet/ina219_without_ethernet.ino`)
- Local network access to the Arduino

## Project Structure

```
arduino/
├── assets/
│   └── ina_diagram.jpg               # Wiring diagram
├── grafana/
│   ├── dashboards/
│   │   └── arduino-energy-dashboard.json
│   └── provisioning/
│       ├── datasources/
│       │   └── datasource.yml
│       └── dashboards/
│           └── dashboard.yml
├── ina219_without_ethernet/
│   └── ina219_without_ethernet.ino   # Arduino sketch
├── prometheus/
│   ├── prometheus.yml
│   └── alerts.yml
├── docker-compose.yaml
├── requirements.txt                 # Python dependencies for the serial collector
├── settings.example.yaml            # Template committed to the repository
├── settings.yaml                     # Local settings (ignored by Git)
├── main.py                           # Python bridge: reads serial → exposes /metrics
└── README.md
```

## Run the serial collector on Windows, Linux, or macOS

The collector reads the Arduino's USB serial output and exposes the readings at `http://localhost:8080/metrics`. The repository includes `settings.example.yaml` as a template; the machine-specific `settings.yaml` is ignored by Git. Before the first run, copy the example to `settings.yaml` and edit `serial_port` for the computer connected to the Arduino. The file is loaded from the same directory as `main.py`; command-line options such as `--port` override its values.

Copy the example from this `arduino/` directory:

```powershell
# Windows (PowerShell)
Copy-Item settings.example.yaml settings.yaml
```

```bash
# Linux or macOS
cp settings.example.yaml settings.yaml
```

Then open `settings.yaml` and set `serial_port` to the port found in the platform-specific instructions below. For example, on the current macOS setup it is `/dev/cu.usbmodem1201`.

Create and activate a virtual environment from this `arduino/` directory, then install the dependencies:

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m serial.tools.list_ports -v
```

Set `serial_port` in `settings.yaml` to the listed port, such as `COM3`, then start the collector:

```powershell
python main.py --mode workload --period tarde
```

**Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m serial.tools.list_ports -v
```

Set `serial_port` to the listed device, commonly `/dev/ttyACM0` or `/dev/ttyUSB0`, then run:

```bash
python main.py --mode workload --period tarde
```

If access is denied, add your user to the serial-device group used by your distribution (commonly `dialout`) and sign in again.

**macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m serial.tools.list_ports -v
```

Set `serial_port` to the listed Arduino device, commonly `/dev/cu.usbmodem...` or `/dev/cu.usbserial...`, then run:

```bash
python main.py --mode workload --period tarde
```

Close Arduino IDE's Serial Monitor before starting the collector so it can take exclusive access to the port. Use `--mode baseline` for an idle measurement; valid periods are `manha`, `tarde`, and `noite`. Replace `workload` and `tarde` with the mode and period for your measurement session.

Install the `pyserial` package listed in `requirements.txt`; the similarly named `serial` package is different and does not provide the `serial.Serial` API used by this script.

## Setup

### 1. Flash the Arduino

Open `ina219_without_ethernet/ina219_without_ethernet.ino` in the Arduino IDE and upload it to your Arduino UNO.

### 2. Configure the Prometheus target

Edit `prometheus/prometheus.yml` and set the correct IP for your machine running `main.py`:

```yaml
scrape_configs:
  - job_name: 'arduino_energia'
    static_configs:
      - targets: ['192.168.0.50:8080']  # <- change to your host IP
```

The serial collector port is configured separately in `settings.yaml`; see the platform-specific instructions above.

### 3. Start the stack

```bash
docker-compose up -d

# Check running containers
docker-compose ps

# Follow logs
docker-compose logs -f
```

### 4. Fix permissions (Linux/macOS only)

```bash
sudo chown -R 65534:65534 prometheus/
sudo chown -R 472:472 grafana/
```

## Accessing the Services

| Service | URL | Credentials |
|---|---|---|
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |

### Verify Prometheus target

Go to http://localhost:9090 → **Status → Targets** — the `arduino_energia` target should show **UP**.

### Open the Grafana dashboard

1. Log in at http://localhost:3000
2. Go to **Dashboards** and open **Arduino Energy Monitor - INA219**

If the dashboard is not auto-provisioned, import it manually:
**+ → Import dashboard → Upload JSON file** → select `grafana/dashboards/arduino-energy-dashboard.json`.

## Useful PromQL Queries

```promql
# Instantaneous current (A)
energia_corrente

# Instantaneous power (W)
energia_potencia

# Instantaneous voltage (V)
energia_tensao

# Average current over the last 5 minutes
avg_over_time(energia_corrente[5m])

# Peak power in the last hour
max_over_time(energia_potencia[1h])

# Estimated total consumption (Wh)
sum(energia_potencia) * 5 / 3600

# Rate of current change
rate(energia_corrente[1m])
```

## Stopping the Stack

```bash
# Stop containers
docker-compose stop

# Stop and remove containers
docker-compose down

# Remove containers and volumes (deletes all stored data)
docker-compose down -v
```

## Updating Configuration

After changing any config file, restart the affected service:

```bash
docker-compose restart
# or fully recreate
docker-compose up -d --force-recreate
```

## Troubleshooting

**Arduino not showing in Prometheus targets**
```bash
ping 192.168.0.50
curl http://192.168.0.50:8080
docker-compose restart prometheus
```

**Grafana shows no data**
1. Go to http://localhost:9090 → Graph → run `energia_corrente`
2. In Grafana go to **Configuration → Data Sources → Prometheus → Test**

**Permission denied errors (Linux/macOS)**
```bash
sudo chown -R $(id -u):$(id -g) .
sudo chmod -R 755 prometheus/ grafana/
```
