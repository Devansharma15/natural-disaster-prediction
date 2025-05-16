import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import os
from utils.constants import (
    USGS_EARTHQUAKE_API,
    NOAA_HURRICANE_API,
    FLOOD_DATA_API
)

def fetch_earthquake_data(time_range):
    """
    Fetch real-time earthquake data from USGS API
    
    Args:
        time_range (str): Time period for which to fetch data ("1 day", "7 days", etc.)
    
    Returns:
        dict: Raw earthquake data
    """
    # Convert time range to starttime parameter
    days = {
        "1 day": 1,
        "7 days": 7,
        "30 days": 30,
        "90 days": 90
    }.get(time_range, 7)
    
    start_time = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    # Set up the API parameters
    params = {
        "format": "geojson",
        "starttime": start_time,
        "minmagnitude": 2.5,
        "orderby": "time"
    }
    
    try:
        # Make the API request
        response = requests.get(USGS_EARTHQUAKE_API, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Basic processing to format the response
            processed_data = {
                'raw_data': data,
                'features': data.get('features', []),
                'metadata': data.get('metadata', {}),
                'total_events': data.get('metadata', {}).get('count', 0)
            }
            
            # Extract basic statistics
            if processed_data['features']:
                magnitudes = [feature['properties']['mag'] for feature in processed_data['features'] 
                              if feature['properties']['mag'] is not None]
                
                if magnitudes:
                    processed_data['max_magnitude'] = max(magnitudes)
                    processed_data['min_magnitude'] = min(magnitudes)
                    processed_data['avg_magnitude'] = sum(magnitudes) / len(magnitudes)
                
                # Create a dataframe for easier processing later
                df_data = []
                for feature in processed_data['features']:
                    props = feature['properties']
                    geo = feature['geometry']['coordinates']
                    
                    df_data.append({
                        'id': props.get('ids', ''),
                        'time': datetime.fromtimestamp(props.get('time', 0)/1000),
                        'updated': datetime.fromtimestamp(props.get('updated', 0)/1000),
                        'magnitude': props.get('mag', None),
                        'place': props.get('place', ''),
                        'location': props.get('place', 'Unknown location'),
                        'depth': geo[2] if len(geo) > 2 else None,
                        'longitude': geo[0] if len(geo) > 0 else None,
                        'latitude': geo[1] if len(geo) > 1 else None,
                        'significance': props.get('sig', 0),
                        'tsunami': props.get('tsunami', 0),
                        'alert': props.get('alert', None)
                    })
                
                processed_data['dataframe'] = pd.DataFrame(df_data)
                
                # Create data for map visualization
                processed_data['data_for_map'] = processed_data['dataframe'][
                    ['latitude', 'longitude', 'magnitude', 'depth', 'place', 'time', 'significance']
                ].copy()
                
                # Create list of active zones
                if not processed_data['dataframe'].empty:
                    active_zones = processed_data['dataframe']['place'].str.split(', ').str[-1].unique()
                    processed_data['active_zones'] = [zone for zone in active_zones if zone]
            
            return processed_data
        else:
            print(f"Error fetching earthquake data: {response.status_code}")
            return {'error': f"Error fetching earthquake data: {response.status_code}"}
    
    except Exception as e:
        print(f"Exception while fetching earthquake data: {e}")
        return {'error': f"Exception while fetching earthquake data: {str(e)}"}

def fetch_hurricane_data(time_range):
    """
    Fetch hurricane data from NOAA API
    
    Args:
        time_range (str): Time period for which to fetch data
    
    Returns:
        dict: Processed hurricane data
    """
    # Convert time range to starttime parameter
    days = {
        "1 day": 1,
        "7 days": 7,
        "30 days": 30,
        "90 days": 90
    }.get(time_range, 7)
    
    try:
        # Make the API request
        # Note: Using a simplified approach here since actual implementation would be more complex
        url = f"{NOAA_HURRICANE_API}?period={days}"
        
        # Simulate an API response for demonstration
        # In a real implementation, you would parse the actual API response
        # We provide detailed structure to demonstrate how the data would be processed
        processed_data = {
            'total_events': 0,
            'active_storms': [],
            'max_category': 'N/A',
            'max_wind_speed': 0,
            'dataframe': pd.DataFrame(),
            'data_for_map': pd.DataFrame()
        }
        
        # Check if we're in hurricane season (June 1 to November 30)
        current_month = datetime.now().month
        is_hurricane_season = 6 <= current_month <= 11
        
        # Generate sample data only during hurricane season or for demonstration
        if is_hurricane_season or days > 30:
            # Create sample hurricane data for visualization
            storm_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
            categories = [1, 2, 3, 4, 5]
            wind_speeds = [75, 96, 111, 130, 157]
            
            # Generate a random number of storms based on time range
            num_storms = min(5, max(1, days // 7))
            
            storm_data = []
            for i in range(num_storms):
                cat_idx = min(i, len(categories) - 1)
                
                # Create multiple positions for each storm to show track
                base_lat = 15 + np.random.uniform(-5, 5)
                base_lon = -60 + np.random.uniform(-10, 10)
                
                for j in range(5):  # 5 positions per storm
                    time_offset = timedelta(hours=j*6)  # 6-hour intervals
                    storm_time = datetime.now() - timedelta(days=np.random.uniform(0, days)) + time_offset
                    
                    # Storms move northwest generally
                    lat_offset = j * 0.5 + np.random.uniform(-0.1, 0.1)
                    lon_offset = j * 0.7 + np.random.uniform(-0.1, 0.1)
                    
                    # Wind speed decreases as storm moves (usually)
                    wind_speed_factor = max(0.7, 1 - (j * 0.05))
                    
                    storm_data.append({
                        'id': f"2023{storm_names[i]}",
                        'name': storm_names[i],
                        'time': storm_time,
                        'latitude': base_lat + lat_offset,
                        'longitude': base_lon + lon_offset,
                        'category': categories[cat_idx],
                        'wind_speed': int(wind_speeds[cat_idx] * wind_speed_factor),
                        'pressure': 1000 - (categories[cat_idx] * 15),
                        'movement_direction': 'NW',
                        'movement_speed': 10 - j,
                    })
            
            if storm_data:
                df = pd.DataFrame(storm_data)
                
                # Sort by time
                df = df.sort_values('time')
                
                # Find the most recent position for each storm
                latest_positions = df.groupby('name').apply(lambda x: x.iloc[-1]).reset_index(drop=True)
                
                processed_data['dataframe'] = df
                processed_data['data_for_map'] = df
                processed_data['active_storms'] = latest_positions['name'].unique().tolist()
                processed_data['total_events'] = len(processed_data['active_storms'])
                processed_data['max_category'] = str(latest_positions['category'].max())
                processed_data['max_wind_speed'] = latest_positions['wind_speed'].max()
        
        return processed_data
    
    except Exception as e:
        print(f"Exception while fetching hurricane data: {e}")
        return {'error': f"Exception while fetching hurricane data: {str(e)}"}

def fetch_flood_data(time_range):
    """
    Fetch flood data from flood monitoring API
    
    Args:
        time_range (str): Time period for which to fetch data
    
    Returns:
        dict: Processed flood data
    """
    # Convert time range to starttime parameter
    days = {
        "1 day": 1,
        "7 days": 7,
        "30 days": 30,
        "90 days": 90
    }.get(time_range, 7)
    
    try:
        # Make the API request
        # Note: Using a simplified approach here since actual implementation would be more complex
        url = f"{FLOOD_DATA_API}?period={days}"
        
        # Simulate an API response for demonstration
        # In a real implementation, you would parse the actual API response
        processed_data = {
            'total_events': 0,
            'affected_areas': [],
            'max_severity': 'N/A',
            'total_area': 0,
            'dataframe': pd.DataFrame(),
            'data_for_map': pd.DataFrame()
        }
        
        # Generate sample flood data for visualization
        flood_areas = [
            {"name": "Mississippi Basin", "country": "USA", "severity": "Moderate", "lat": 39.166667, "lon": -94.383333},
            {"name": "Mekong Delta", "country": "Vietnam", "severity": "Severe", "lat": 9.823333, "lon": 105.799722},
            {"name": "Ganges Delta", "country": "Bangladesh", "severity": "Severe", "lat": 23.0, "lon": 90.0},
            {"name": "Rhine Valley", "country": "Germany", "severity": "Mild", "lat": 51.233333, "lon": 6.783333},
            {"name": "Murray-Darling Basin", "country": "Australia", "severity": "Moderate", "lat": -34.9, "lon": 142.2},
        ]
        
        # Filter based on time range - longer time range means more floods
        num_floods = min(len(flood_areas), max(1, days // 10))
        sample_floods = flood_areas[:num_floods]
        
        flood_data = []
        severity_map = {"Mild": 1, "Moderate": 2, "Severe": 3}
        
        for area in sample_floods:
            # Generate several data points per flood area
            for i in range(3):  # 3 data points per area
                flood_time = datetime.now() - timedelta(days=np.random.uniform(0, days))
                
                # Small variations for each data point
                lat_offset = np.random.uniform(-0.5, 0.5)
                lon_offset = np.random.uniform(-0.5, 0.5)
                
                # Area affected (in square km)
                area_affected = 100 * severity_map[area["severity"]] * np.random.uniform(0.8, 1.2)
                
                flood_data.append({
                    'id': f"{area['name']}-{i}",
                    'name': area['name'],
                    'country': area['country'],
                    'time': flood_time,
                    'latitude': area['lat'] + lat_offset,
                    'longitude': area['lon'] + lon_offset,
                    'severity': area['severity'],
                    'severity_value': severity_map[area['severity']],
                    'area_affected': area_affected,
                    'population_affected': int(area_affected * 100 * np.random.uniform(0.5, 1.5)),
                    'water_level_change': severity_map[area['severity']] * np.random.uniform(0.8, 1.2),
                })
        
        if flood_data:
            df = pd.DataFrame(flood_data)
            
            # Sort by time
            df = df.sort_values('time')
            
            processed_data['dataframe'] = df
            processed_data['data_for_map'] = df
            processed_data['affected_areas'] = df['name'].unique().tolist()
            processed_data['total_events'] = len(df)
            processed_data['max_severity'] = df['severity_value'].max()
            processed_data['total_area'] = int(df['area_affected'].sum())
        
        return processed_data
    
    except Exception as e:
        print(f"Exception while fetching flood data: {e}")
        return {'error': f"Exception while fetching flood data: {str(e)}"}

def fetch_historical_disasters(disaster_type, time_range=None, region=None):
    """
    Fetch historical disaster data for analysis
    
    Args:
        disaster_type (str): Type of disaster (Earthquake, Hurricane, Flood)
        time_range (str, optional): Time period for filtering
        region (str, optional): Region for filtering
    
    Returns:
        pandas.DataFrame: Historical disaster data
    """
    # For demonstration, we'll generate synthetic historical data
    # In a real implementation, this would come from a database or historical API
    
    # Generate date range for the past 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)  # 5 years of data
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='W')  # Weekly data points
    
    if disaster_type == "Earthquake":
        # Generate earthquake data with seasonal patterns
        magnitudes = []
        for date in date_range:
            # Baseline magnitude with some randomness
            base_magnitude = 5.0 + np.random.normal(0, 0.5)
            # No strong seasonal pattern for earthquakes, but add some variation
            month_factor = 0.1 * np.sin(date.month * np.pi / 6)
            magnitudes.append(max(2.5, base_magnitude + month_factor))
        
        df = pd.DataFrame({
            'date': date_range,
            'magnitude': magnitudes,
            'depth': np.random.uniform(5, 30, size=len(date_range)),
            'region': np.random.choice(['Pacific Ring', 'Mediterranean', 'Himalayan', 'Other'], size=len(date_range)),
            'casualties': np.random.exponential(10, size=len(date_range)).astype(int)
        })
        
        # Add some major events
        major_events = pd.DataFrame({
            'date': [
                datetime(2019, 7, 6),  # Southern California earthquake
                datetime(2020, 3, 22),  # Croatia earthquake
                datetime(2021, 8, 14),  # Haiti earthquake
                datetime(2022, 11, 21),  # Indonesia earthquake
            ],
            'magnitude': [7.1, 5.3, 7.2, 5.6],
            'depth': [8, 10, 10, 10],
            'region': ['Pacific Ring', 'Mediterranean', 'Caribbean', 'Pacific Ring'],
            'casualties': [0, 1, 2248, 300]
        })
        
        df = pd.concat([df, major_events]).sort_values('date').reset_index(drop=True)
        
    elif disaster_type == "Hurricane":
        # Generate hurricane data with strong seasonal pattern
        wind_speeds = []
        counts = []
        
        for date in date_range:
            # Hurricane season is roughly June through November
            is_season = 6 <= date.month <= 11
            
            # Season factor: higher during hurricane season
            season_factor = 50 if is_season else 10
            
            # Wind speed has seasonal pattern
            base_speed = np.random.normal(season_factor, 10)
            wind_speeds.append(max(0, base_speed))
            
            # Count of active hurricanes also follows seasonal pattern
            count_base = 3 if is_season else 0
            counts.append(max(0, int(np.random.normal(count_base, 1))))
        
        df = pd.DataFrame({
            'date': date_range,
            'wind_speed': wind_speeds,
            'active_count': counts,
            'region': np.random.choice(['Atlantic', 'Pacific', 'Indian', 'Other'], size=len(date_range)),
            'damage_millions': np.random.exponential(100, size=len(date_range))
        })
        
        # Add some major events
        major_events = pd.DataFrame({
            'date': [
                datetime(2019, 9, 1),  # Hurricane Dorian
                datetime(2020, 8, 27),  # Hurricane Laura
                datetime(2021, 8, 29),  # Hurricane Ida
                datetime(2022, 9, 28),  # Hurricane Ian
            ],
            'wind_speed': [185, 150, 150, 155],
            'active_count': [1, 1, 1, 1],
            'region': ['Atlantic', 'Atlantic', 'Atlantic', 'Atlantic'],
            'damage_millions': [3400, 19000, 75000, 50000]
        })
        
        df = pd.concat([df, major_events]).sort_values('date').reset_index(drop=True)
        
    elif disaster_type == "Flood":
        # Generate flood data with seasonal patterns
        severities = []
        areas = []
        
        for date in date_range:
            # Many regions have rainy seasons in spring/summer
            month_factor = np.sin(date.month * np.pi / 6)  # Peak in July
            
            # Severity follows seasonal pattern
            base_severity = 1 + 1.5 * month_factor + np.random.normal(0, 0.5)
            severities.append(max(0, min(3, base_severity)))  # Scale 0-3
            
            # Area affected also follows seasonal pattern
            area_base = 1000 * (1 + month_factor) + np.random.exponential(500)
            areas.append(max(0, area_base))
        
        df = pd.DataFrame({
            'date': date_range,
            'severity_value': severities,
            'area_affected': areas,
            'region': np.random.choice(['Asia', 'Europe', 'North America', 'Africa', 'South America'], size=len(date_range)),
            'casualties': np.random.exponential(5, size=len(date_range)).astype(int)
        })
        
        # Add some major events
        major_events = pd.DataFrame({
            'date': [
                datetime(2019, 3, 15),  # Midwest US floods
                datetime(2020, 7, 7),   # China floods
                datetime(2021, 7, 14),  # European floods
                datetime(2022, 8, 30),  # Pakistan floods
            ],
            'severity_value': [2.8, 2.9, 2.7, 3.0],
            'area_affected': [8000, 12000, 7000, 30000],
            'region': ['North America', 'Asia', 'Europe', 'Asia'],
            'casualties': [10, 140, 242, 1700]
        })
        
        df = pd.concat([df, major_events]).sort_values('date').reset_index(drop=True)
    
    else:
        # Default empty dataframe
        df = pd.DataFrame(columns=['date', 'value', 'region'])
    
    # Apply filters if provided
    if region and region != "Global":
        # Map region filter to dataframe regions (simplified mapping)
        region_mapping = {
            "North America": ["North America", "Caribbean"],
            "South America": ["South America"],
            "Europe": ["Europe", "Mediterranean"],
            "Asia": ["Asia", "Himalayan", "Pacific Ring"],
            "Africa": ["Africa"],
            "Oceania": ["Pacific", "Other"]
        }
        
        if region in region_mapping:
            df = df[df['region'].isin(region_mapping[region])]
    
    if time_range:
        days = {
            "1 day": 1,
            "7 days": 7,
            "30 days": 30,
            "90 days": 90
        }.get(time_range, 365 * 5)  # Default to all 5 years
        
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['date'] >= cutoff_date]
    
    return df

def fetch_sample_data(disaster_type):
    """
    Fetch sample data for model testing
    
    Args:
        disaster_type (str): Type of disaster (Earthquake, Hurricane, Flood)
    
    Returns:
        pandas.DataFrame: Sample data for testing models
    """
    # Generate sample data for model testing
    if disaster_type == "Earthquake":
        # Create sample earthquake data
        num_samples = 1000
        
        # Generate parameters
        latitudes = np.random.uniform(-60, 70, num_samples)
        longitudes = np.random.uniform(-180, 180, num_samples)
        depths = np.random.gamma(shape=5, scale=3, size=num_samples)  # Most earthquakes are shallow
        
        # Historical time period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)  # 2 years
        dates = [start_date + timedelta(days=x) for x in np.random.randint(0, 730, num_samples)]
        
        # Generate magnitude with spatial and depth correlations
        # Higher magnitudes tend to occur at specific locations (tectonically active areas)
        base_magnitudes = np.zeros(num_samples)
        
        # Pacific Ring of Fire (higher magnitudes)
        pacific_rim_mask = (
            ((longitudes > 150) | (longitudes < -120)) & 
            (latitudes > -50) & (latitudes < 60)
        )
        base_magnitudes[pacific_rim_mask] += 1.0
        
        # Himalayan belt (higher magnitudes)
        himalayan_mask = (
            (longitudes > 60) & (longitudes < 100) & 
            (latitudes > 25) & (latitudes < 40)
        )
        base_magnitudes[himalayan_mask] += 0.8
        
        # Mediterranean-Alpine belt (medium magnitudes)
        mediterranean_mask = (
            (longitudes > -10) & (longitudes < 40) & 
            (latitudes > 30) & (latitudes < 45)
        )
        base_magnitudes[mediterranean_mask] += 0.5
        
        # Add depth correlation (deeper quakes often have different magnitude distributions)
        base_magnitudes += depths * 0.02
        
        # Add randomness
        magnitudes = base_magnitudes + np.random.normal(4, 0.7, num_samples)
        
        # Ensure valid magnitude range (2.5+)
        magnitudes = np.maximum(magnitudes, 2.5)
        
        # Create the DataFrame
        df = pd.DataFrame({
            'latitude': latitudes,
            'longitude': longitudes,
            'depth': depths,
            'magnitude': magnitudes,
            'time': dates,
            'region': ['Sample Region'] * num_samples,
            'has_tsunami': np.random.choice([0, 1], size=num_samples, p=[0.95, 0.05]),
            'significance': np.random.randint(0, 1000, num_samples)
        })
        
        return df
        
    elif disaster_type == "Hurricane":
        # Create sample hurricane data
        num_storms = 50
        positions_per_storm = 20
        
        storm_data = []
        storm_names = [f"Storm_{i}" for i in range(num_storms)]
        
        # End date is now, start date is 3 years ago
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)
        
        for i, name in enumerate(storm_names):
            # Storm starting location (typically Atlantic or Pacific for hurricanes)
            basin = np.random.choice(['Atlantic', 'Pacific'])
            
            if basin == 'Atlantic':
                start_lat = np.random.uniform(10, 20)
                start_lon = np.random.uniform(-60, -30)
            else:  # Pacific
                start_lat = np.random.uniform(10, 20)
                start_lon = np.random.uniform(120, 160)
            
            # Storm starting time (mostly during hurricane season: June-November)
            month = np.random.choice([6, 7, 8, 9, 10, 11])
            year = np.random.randint(start_date.year, end_date.year + 1)
            day = np.random.randint(1, 28)
            storm_start = datetime(year, month, day)
            
            # Make sure it's within our timeframe
            if storm_start < start_date:
                storm_start = start_date
            if storm_start > end_date:
                storm_start = end_date - timedelta(days=30)
            
            # Storm initial intensity
            initial_category = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.15, 0.05])
            initial_wind = 75 + (initial_category - 1) * 20 + np.random.normal(0, 5)
            
            # Generate storm track
            for j in range(positions_per_storm):
                time_offset = j * 6  # 6-hour intervals
                current_time = storm_start + timedelta(hours=time_offset)
                
                # Storm moves generally northwest then northeast
                if j < positions_per_storm // 2:
                    # Northwestern movement
                    lat_offset = j * 0.5 + np.random.normal(0, 0.1)
                    lon_offset = j * -0.7 + np.random.normal(0, 0.1)
                else:
                    # Northeastern recurve
                    lat_offset = positions_per_storm // 2 * 0.5 + (j - positions_per_storm // 2) * 0.6 + np.random.normal(0, 0.1)
                    lon_offset = positions_per_storm // 2 * -0.7 + (j - positions_per_storm // 2) * 0.4 + np.random.normal(0, 0.1)
                
                current_lat = start_lat + lat_offset
                current_lon = start_lon + lon_offset
                
                # Storm intensity changes over time (typically strengthens then weakens)
                intensity_factor = 1.0
                if j < positions_per_storm // 3:
                    # Strengthening phase
                    intensity_factor = 1.0 + (j / (positions_per_storm // 3)) * 0.5
                else:
                    # Weakening phase
                    intensity_factor = 1.5 - ((j - positions_per_storm // 3) / (positions_per_storm * 2/3)) * 1.0
                
                current_wind = initial_wind * intensity_factor + np.random.normal(0, 5)
                
                # Determine category based on wind speed
                if current_wind < 75:
                    category = 0  # Tropical Storm
                elif current_wind < 96:
                    category = 1
                elif current_wind < 111:
                    category = 2
                elif current_wind < 130:
                    category = 3
                elif current_wind < 157:
                    category = 4
                else:
                    category = 5
                
                storm_data.append({
                    'storm_id': i,
                    'name': name,
                    'basin': basin,
                    'time': current_time,
                    'latitude': current_lat,
                    'longitude': current_lon,
                    'wind_speed': current_wind,
                    'category': category,
                    'pressure': 1010 - (current_wind * 0.75) + np.random.normal(0, 3),
                    'position_number': j
                })
        
        # Create dataframe
        df = pd.DataFrame(storm_data)
        return df
        
    elif disaster_type == "Flood":
        # Create sample flood data
        num_samples = 500
        
        # Generate parameters
        latitudes = np.random.uniform(-50, 60, num_samples)
        longitudes = np.random.uniform(-180, 180, num_samples)
        
        # Historical time period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)  # 3 years
        dates = [start_date + timedelta(days=x) for x in np.random.randint(0, 1095, num_samples)]
        
        # Generate flood severity with seasonal and geographical correlation
        base_severities = np.zeros(num_samples)
        
        # Add seasonal pattern - most floods happen in specific seasons
        for i, date in enumerate(dates):
            # Season factor: higher during typical flood seasons (varies by region)
            month_factor = np.sin((date.month - 3) * np.pi / 6)  # Peak around June/July in Northern Hemisphere
            base_severities[i] += max(0, month_factor)
        
        # Add geographical patterns
        # Major river basins and coastal areas are more prone to flooding
        
        # Low-lying coastal areas
        coastal_mask = (
            (np.abs(latitudes) < 30) & 
            ((np.abs(longitudes) > 160) | (np.abs(longitudes) < 20))
        )
        base_severities[coastal_mask] += 0.7
        
        # Major river basins
        # Amazon basin
        amazon_mask = (
            (longitudes > -80) & (longitudes < -50) & 
            (latitudes > -10) & (latitudes < 5)
        )
        base_severities[amazon_mask] += 0.8
        
        # Ganges-Brahmaputra basin
        ganges_mask = (
            (longitudes > 75) & (longitudes < 95) & 
            (latitudes > 20) & (latitudes < 30)
        )
        base_severities[ganges_mask] += 0.9
        
        # Mississippi basin
        mississippi_mask = (
            (longitudes > -100) & (longitudes < -80) & 
            (latitudes > 30) & (latitudes < 45)
        )
        base_severities[mississippi_mask] += 0.6
        
        # Add randomness
        severities = base_severities + np.random.normal(1, 0.4, num_samples)
        
        # Ensure valid severity range (0-3 scale)
        severities = np.maximum(0, np.minimum(severities, 3))
        
        # Derive other parameters based on severity
        precipitation = 50 + severities * 100 + np.random.normal(0, 20, num_samples)
        water_level = 2 + severities * 3 + np.random.normal(0, 0.5, num_samples)
        area_affected = severities * 1000 + np.random.exponential(500, num_samples)
        
        # Create the DataFrame
        df = pd.DataFrame({
            'latitude': latitudes,
            'longitude': longitudes,
            'time': dates,
            'severity_value': severities,
            'precipitation_mm': precipitation,
            'water_level_m': water_level,
            'area_affected_km2': area_affected,
            'population_affected': (area_affected * 50 + np.random.exponential(1000, num_samples)).astype(int),
            'duration_days': (severities * 3 + np.random.exponential(2, num_samples)).astype(int),
            'region': ['Sample Region'] * num_samples
        })
        
        return df
    
    # Default empty dataframe
    return pd.DataFrame()
