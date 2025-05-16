"""
Constants used by the natural disaster prediction system
"""

# API Endpoints
USGS_EARTHQUAKE_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"
NOAA_HURRICANE_API = "https://www.nhc.noaa.gov/data/api/archive"  # Example endpoint
FLOOD_DATA_API = "https://global.floods.rti.org/api/events"  # Example endpoint

# Time ranges for data fetching
TIME_RANGES = {
    "1 day": 1,
    "7 days": 7,
    "30 days": 30,
    "90 days": 90
}

# Region definitions and boundaries
REGIONS = {
    "Global": {
        "label": "Global (All Regions)",
        "bounds": None
    },
    "North America": {
        "label": "North America",
        "bounds": {
            "lat": (15, 90),
            "lon": (-170, -30)
        }
    },
    "South America": {
        "label": "South America",
        "bounds": {
            "lat": (-60, 15),
            "lon": (-90, -30)
        }
    },
    "Europe": {
        "label": "Europe",
        "bounds": {
            "lat": (35, 75),
            "lon": (-25, 40)
        }
    },
    "Asia": {
        "label": "Asia",
        "bounds": {
            "lat": (0, 80),
            "lon": (40, 180)
        }
    },
    "Africa": {
        "label": "Africa",
        "bounds": {
            "lat": (-40, 40),
            "lon": (-20, 55)
        }
    },
    "Oceania": {
        "label": "Oceania",
        "bounds": {
            "lat": (-50, 0),
            "lon": (100, 180)
        }
    }
}

# Model configuration constants
PREDICTION_MODELS = {
    "Random Forest": {
        "description": "Ensemble learning method using multiple decision trees",
        "parameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
    },
    "LSTM": {
        "description": "Long Short-Term Memory neural network for time series",
        "parameters": {
            "units": 50,
            "dropout": 0.2,
            "recurrent_dropout": 0.2
        }
    },
    "Prophet": {
        "description": "Facebook's time series forecasting model",
        "parameters": {
            "seasonality_mode": "multiplicative",
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False
        }
    }
}

# Disaster-specific constants
EARTHQUAKE_MAGNITUDE_SCALE = {
    "Minor": {"range": (0, 4.0), "color": "#1a9850"},
    "Light": {"range": (4.0, 5.0), "color": "#91cf60"},
    "Moderate": {"range": (5.0, 6.0), "color": "#fee08b"},
    "Strong": {"range": (6.0, 7.0), "color": "#fc8d59"},
    "Major": {"range": (7.0, float('inf')), "color": "#d73027"}
}

HURRICANE_CATEGORY_SCALE = {
    "Tropical Depression": {"wind_speed": (0, 39), "color": "#bdd7e7"},
    "Tropical Storm": {"wind_speed": (39, 74), "color": "#6baed6"},
    "Category 1": {"wind_speed": (74, 96), "color": "#3182bd"},
    "Category 2": {"wind_speed": (96, 111), "color": "#08519c"},
    "Category 3": {"wind_speed": (111, 130), "color": "#b30000"},
    "Category 4": {"wind_speed": (130, 157), "color": "#7f0000"},
    "Category 5": {"wind_speed": (157, float('inf')), "color": "#4a0000"}
}

FLOOD_SEVERITY_SCALE = {
    "Minor": {"range": (0, 1.0), "color": "#a6cee3"},
    "Moderate": {"range": (1.0, 2.0), "color": "#1f78b4"},
    "Severe": {"range": (2.0, float('inf')), "color": "#08306b"}
}

# Data refresh intervals (in seconds)
DEFAULT_REFRESH_INTERVAL = 21600  # 6 hours

# Default confidence threshold for alerts
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# Emergency contact information
EMERGENCY_CONTACTS = {
    "Global Disaster Alert": {
        "name": "Global Disaster Alert and Coordination System",
        "url": "https://www.gdacs.org/",
        "phone": "N/A"
    },
    "Red Cross": {
        "name": "International Red Cross",
        "url": "https://www.redcross.org/",
        "phone": "+1-800-RED-CROSS"
    },
    "FEMA": {
        "name": "Federal Emergency Management Agency",
        "url": "https://www.fema.gov/",
        "phone": "+1-800-621-FEMA"
    }
}

# Data source information
DATA_SOURCES = {
    "Earthquake": {
        "name": "USGS Earthquake Catalog",
        "url": "https://earthquake.usgs.gov/earthquakes/feed/",
        "description": "Real-time and historical earthquake data from the United States Geological Survey"
    },
    "Hurricane": {
        "name": "NOAA National Hurricane Center",
        "url": "https://www.nhc.noaa.gov/data/",
        "description": "Hurricane and tropical storm data from the National Oceanic and Atmospheric Administration"
    },
    "Flood": {
        "name": "Global Flood Monitoring",
        "url": "https://global.floods.rti.org/",
        "description": "Global flood monitoring data from satellite and ground observations"
    }
}
