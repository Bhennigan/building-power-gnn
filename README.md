# Building Power Efficiency GNN

A Graph Neural Network system for predictive power efficiency analysis in buildings, with intuitive data ingestion and live API integrations.

## Features

- **Graph Neural Network Model**: Heterogeneous GNN architecture for building energy prediction
- **Flexible Data Ingestion**: Upload building data via CSV, Excel, JSON, or XML
- **Live Data Integrations**: Connect to external APIs for real-time data
  - Project Haystack (smart building semantic data)
  - Green Button (utility energy usage data)
  - Weather APIs (Open-Meteo, OpenWeatherMap, NOAA)
  - Power Monitors (Emporia, IoTaWatt, Shelly, Home Assistant)
- **Address-Based Weather**: Automatic geocoding of building addresses for weather data
- **Multi-Tenant Architecture**: Secure isolation between organizations
- **Web Interface**: Modern UI for managing buildings, data, and integrations

## Architecture

```
+------------------+     +------------------+     +------------------+
|  Data Ingestion  | --> |  Graph Builder   | --> |   GNN Model      |
|  (CSV/XML/JSON)  |     |  (NetworkX/PyG)  |     |  (PyTorch)       |
+------------------+     +------------------+     +------------------+
         |                        |                        |
         v                        v                        v
+------------------+     +------------------+     +------------------+
|  Live APIs       |     |  Feature Store   |     |  Prediction API  |
|  (Haystack/etc)  |     |  (Time-series)   |     |  (FastAPI)       |
+------------------+     +------------------+     +------------------+
```

## Graph Structure

### Node Types
| Type | Description | Example Attributes |
|------|-------------|-------------------|
| HVAC | Climate control equipment | capacity_kw, efficiency |
| Lighting | Lighting systems | wattage, type |
| Sensor | Monitoring devices | reading_type, accuracy |
| Room | Physical spaces | area_sqft, zone |
| Meter | Power meters | type, resolution |
| Equipment | General equipment | power_rating |

### Edge Types
| Type | Description |
|------|-------------|
| serves | Equipment serves a space |
| monitors | Sensor monitors a space |
| feeds | Power flows between nodes |
| adjacent | Spatial proximity |
| controls | Control relationship |
| contains | Hierarchical containment |

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/Bhennigan/building-power-gnn.git
cd building-power-gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from src.db.session import init_db; init_db()"

# Run the server
python -m src.api.main
```

The server will start at http://127.0.0.1:8082

## Usage

### Web Interface

1. **Register/Login**: Create an account at `/register`
2. **Create Building**: Add buildings with address information
3. **Upload Data**: Import nodes, edges, and readings via CSV/JSON/XML
4. **Add Integrations**: Connect to live data sources (weather, Haystack, etc.)
5. **View Insights**: Analyze predictions and efficiency metrics

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/auth/register` | POST | Register new user |
| `/api/v1/auth/login` | POST | Login |
| `/api/v1/buildings` | GET/POST | List/create buildings |
| `/api/v1/buildings/{id}/upload` | POST | Upload building data |
| `/api/v1/integrations` | GET/POST | List/create integrations |
| `/api/v1/integrations/weather` | POST | Create weather integration |
| `/api/v1/integrations/power-monitor` | POST | Create power monitor integration |
| `/api/v1/integrations/{id}/test` | POST | Test integration connection |
| `/api/v1/integrations/{id}/sync` | POST | Sync integration data |
| `/api/v1/integrations/{id}/live` | GET | Get live power readings |
| `/api/v1/integrations/{id}/devices` | GET | Discover power monitor devices |
| `/api/v1/integrations/{id}/channels` | GET | Get monitoring channels |
| `/api/v1/predict/{building_id}` | POST | Get predictions |

### Data Upload Formats

#### CSV Format
```csv
node_id,node_type,name,zone
hvac_1,HVAC,Main HVAC,floor_1
sensor_1,Sensor,Temp Sensor,floor_1
```

#### JSON Format
```json
{
  "nodes": [
    {"id": "hvac_1", "type": "HVAC", "name": "Main HVAC", "zone": "floor_1"}
  ],
  "edges": [
    {"source": "hvac_1", "target": "room_1", "type": "serves"}
  ],
  "readings": [
    {"node_id": "sensor_1", "timestamp": "2024-01-01T08:00:00", "value": 72.5}
  ]
}
```

#### XML Format
```xml
<BuildingGraph>
  <Nodes>
    <Node id="hvac_1" type="HVAC" name="Main HVAC" zone="floor_1"/>
  </Nodes>
  <Edges>
    <Edge source="hvac_1" target="room_1" type="serves"/>
  </Edges>
</BuildingGraph>
```

## Integrations

### Weather Integration

Weather data can be fetched using:
- **Building Address**: Automatically geocodes the building's street address
- **Direct Address**: Provide any address to geocode
- **Coordinates**: Manual latitude/longitude input

Supported providers:
- **Open-Meteo** (free, no API key required)
- **OpenWeatherMap** (requires API key)
- **NOAA Weather.gov** (US locations)

### Project Haystack

Connect to Haystack-compatible building automation systems:
- Automatic point discovery
- Historical data sync
- Real-time readings

### Green Button

Import utility energy usage data via the ESPI standard.

### Power Monitors

Connect to real-time energy monitoring devices for live power consumption data.

#### Supported Devices

| Provider | Type | Authentication | Features |
|----------|------|----------------|----------|
| **Emporia Vue** | Cloud | Email/password | Whole-home monitoring, circuit-level data, solar tracking |
| **IoTaWatt** | Local | None/optional | Open-source, up to 14 circuits, local storage |
| **Shelly EM** | Local | None | WiFi energy meter, 2 channels, contactor control |
| **Home Assistant** | Local/Cloud | Bearer token | Aggregates any HA energy sensor |
| **Generic REST** | Any | Configurable | Custom API endpoints and data mapping |

#### Emporia Setup

```json
{
  "name": "Home Energy Monitor",
  "provider": "emporia",
  "email": "your@email.com",
  "password": "your-password",
  "sync_interval_minutes": 5
}
```

#### Local Device Setup (IoTaWatt/Shelly)

```json
{
  "name": "IoTaWatt Monitor",
  "provider": "iotawatt",
  "base_url": "http://192.168.1.100",
  "sync_interval_minutes": 1
}
```

#### Home Assistant Setup

```json
{
  "name": "Home Assistant Energy",
  "provider": "home_assistant",
  "base_url": "http://homeassistant.local:8123",
  "api_key": "your-long-lived-access-token",
  "sync_interval_minutes": 5
}
```

#### Live Readings API

Get real-time power consumption:

```bash
GET /api/v1/integrations/{id}/live

# Response
[
  {
    "channel_id": "device_1",
    "channel_name": "Main Panel",
    "watts": 3245.5,
    "voltage": 121.2,
    "timestamp": "2024-01-15T10:30:00Z"
  }
]
```

## Project Structure

```
building-power-gnn/
├── src/
│   ├── api/              # FastAPI application
│   │   ├── routes/       # API endpoints
│   │   └── websocket.py  # Real-time updates
│   ├── auth/             # Authentication
│   ├── db/               # Database models and CRUD
│   ├── graph/            # Graph construction
│   ├── ingestion/        # Data parsing
│   ├── integrations/     # External API connectors
│   └── model/            # GNN architecture
├── templates/            # HTML templates
├── static/               # Static assets
└── schemas/              # XSD validation schemas
```

## Configuration

Environment variables:
- `DATABASE_URL`: Database connection string (default: SQLite)
- `SECRET_KEY`: JWT signing key
- `DEBUG`: Enable debug mode

## License

MIT License
