import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def preprocess_earthquake_data(data, region=None):
    """
    Preprocess raw earthquake data for visualization and analysis
    
    Args:
        data (dict): Raw earthquake data from API
        region (str, optional): Region filter to apply
    
    Returns:
        dict: Processed earthquake data
    """
    # Check if data is valid
    if not data or 'error' in data:
        return {'error': data.get('error', 'Unknown error in earthquake data')}
    
    # Extract the dataframe if it exists
    if 'dataframe' not in data:
        return {'error': 'No dataframe found in earthquake data'}
    
    df = data['dataframe']
    
    if df.empty:
        return {'error': 'Empty dataframe in earthquake data'}
    
    # Apply region filter if specified
    if region and region != "Global":
        df = _filter_by_region(df, region)
        
        if df.empty:
            return {'error': f'No earthquake data found for region: {region}'}
    
    # Calculate additional statistics
    result = {
        'dataframe': df,
        'total_events': len(df),
        'data_for_map': df[['latitude', 'longitude', 'magnitude', 'depth', 'place', 'time', 'significance']].copy() 
                          if all(col in df.columns for col in ['latitude', 'longitude', 'magnitude']) else pd.DataFrame()
    }
    
    # Calculate magnitude statistics if available
    if 'magnitude' in df.columns:
        result['max_magnitude'] = df['magnitude'].max()
        result['min_magnitude'] = df['magnitude'].min()
        result['avg_magnitude'] = df['magnitude'].mean()
    
    # Identify active zones
    if 'place' in df.columns:
        # Extract region from place description
        regions = df['place'].str.split(', ').str[-1].value_counts()
        result['active_zones'] = regions.index.tolist()
    
    # Return the processed data
    return result

def preprocess_hurricane_data(data, region=None):
    """
    Preprocess raw hurricane data for visualization and analysis
    
    Args:
        data (dict): Raw hurricane data from API
        region (str, optional): Region filter to apply
    
    Returns:
        dict: Processed hurricane data
    """
    # Check if data is valid
    if not data or 'error' in data:
        return {'error': data.get('error', 'Unknown error in hurricane data')}
    
    # Extract the dataframe if it exists
    if 'dataframe' not in data:
        return {'error': 'No dataframe found in hurricane data'}
    
    df = data['dataframe']
    
    if df.empty:
        return {'error': 'Empty dataframe in hurricane data'}
    
    # Apply region filter if specified
    if region and region != "Global":
        df = _filter_by_region(df, region)
        
        if df.empty:
            return {'error': f'No hurricane data found for region: {region}'}
    
    # Calculate additional statistics
    result = {
        'dataframe': df,
        'data_for_map': df,
        'total_events': len(df['name'].unique()) if 'name' in df.columns else 0
    }
    
    # Find active storms (unique storm names)
    if 'name' in df.columns:
        result['active_storms'] = df['name'].unique().tolist()
    
    # Calculate wind speed statistics if available
    if 'wind_speed' in df.columns:
        result['max_wind_speed'] = df['wind_speed'].max()
        result['min_wind_speed'] = df['wind_speed'].min()
        result['avg_wind_speed'] = df['wind_speed'].mean()
    
    # Find maximum category if available
    if 'category' in df.columns:
        result['max_category'] = str(df['category'].max())
    
    # Return the processed data
    return result

def preprocess_flood_data(data, region=None):
    """
    Preprocess raw flood data for visualization and analysis
    
    Args:
        data (dict): Raw flood data from API
        region (str, optional): Region filter to apply
    
    Returns:
        dict: Processed flood data
    """
    # Check if data is valid
    if not data or 'error' in data:
        return {'error': data.get('error', 'Unknown error in flood data')}
    
    # Extract the dataframe if it exists
    if 'dataframe' not in data:
        return {'error': 'No dataframe found in flood data'}
    
    df = data['dataframe']
    
    if df.empty:
        return {'error': 'Empty dataframe in flood data'}
    
    # Apply region filter if specified
    if region and region != "Global":
        df = _filter_by_region(df, region)
        
        if df.empty:
            return {'error': f'No flood data found for region: {region}'}
    
    # Calculate additional statistics
    result = {
        'dataframe': df,
        'data_for_map': df,
        'total_events': len(df)
    }
    
    # Find affected areas
    if 'name' in df.columns:
        result['affected_areas'] = df['name'].unique().tolist()
    elif 'country' in df.columns:
        result['affected_areas'] = df['country'].unique().tolist()
    
    # Calculate severity statistics if available
    if 'severity_value' in df.columns:
        result['max_severity'] = df['severity_value'].max()
        result['min_severity'] = df['severity_value'].min()
        result['avg_severity'] = df['severity_value'].mean()
    
    # Calculate area statistics if available
    if 'area_affected_km2' in df.columns:
        result['total_area'] = int(df['area_affected_km2'].sum())
    elif 'area_affected' in df.columns:
        result['total_area'] = int(df['area_affected'].sum())
    
    # Return the processed data
    return result

def _filter_by_region(df, region):
    """
    Filter dataframe by geographical region
    
    Args:
        df (pandas.DataFrame): DataFrame to filter
        region (str): Region name to filter by
    
    Returns:
        pandas.DataFrame: Filtered dataframe
    """
    # Define region boundaries (approximate)
    region_bounds = {
        "North America": {
            "lat": (15, 90),
            "lon": (-170, -30)
        },
        "South America": {
            "lat": (-60, 15),
            "lon": (-90, -30)
        },
        "Europe": {
            "lat": (35, 75),
            "lon": (-25, 40)
        },
        "Asia": {
            "lat": (0, 80),
            "lon": (40, 180)
        },
        "Africa": {
            "lat": (-40, 40),
            "lon": (-20, 55)
        },
        "Oceania": {
            "lat": (-50, 0),
            "lon": (100, 180)
        }
    }
    
    # If region is not in our defined regions, return the original dataframe
    if region not in region_bounds:
        return df
    
    # Get region boundaries
    bounds = region_bounds[region]
    
    # Filter by latitude and longitude if available
    if 'latitude' in df.columns and 'longitude' in df.columns:
        filtered_df = df[
            (df['latitude'] >= bounds["lat"][0]) & 
            (df['latitude'] <= bounds["lat"][1]) & 
            (df['longitude'] >= bounds["lon"][0]) & 
            (df['longitude'] <= bounds["lon"][1])
        ]
        return filtered_df
    
    # If no coordinates available but 'region' column exists, try to match by name
    elif 'region' in df.columns:
        # Map region names (simplistic approach)
        region_keywords = {
            "North America": ["north america", "usa", "canada", "mexico", "caribbean"],
            "South America": ["south america", "brazil", "argentina", "chile", "peru"],
            "Europe": ["europe", "eu", "uk", "germany", "france", "italy", "spain"],
            "Asia": ["asia", "china", "japan", "india", "russia", "middle east"],
            "Africa": ["africa", "egypt", "nigeria", "south africa", "kenya"],
            "Oceania": ["oceania", "australia", "new zealand", "pacific"]
        }
        
        keywords = region_keywords.get(region, [])
        if not keywords:
            return df
        
        # Filter by region keywords
        mask = df['region'].str.lower().apply(lambda x: any(kw in str(x).lower() for kw in keywords))
        filtered_df = df[mask]
        return filtered_df
    
    # If neither coordinates nor region column is available, return original dataframe
    return df

def process_time_range(time_range):
    """
    Convert time range string to datetime objects
    
    Args:
        time_range (str): Time range string (e.g., "7 days")
    
    Returns:
        tuple: (start_date, end_date)
    """
    end_date = datetime.now()
    
    # Parse time range
    if time_range == "1 day":
        start_date = end_date - timedelta(days=1)
    elif time_range == "7 days":
        start_date = end_date - timedelta(days=7)
    elif time_range == "30 days":
        start_date = end_date - timedelta(days=30)
    elif time_range == "90 days":
        start_date = end_date - timedelta(days=90)
    else:
        # Default to 7 days
        start_date = end_date - timedelta(days=7)
    
    return start_date, end_date

def enrich_earthquake_data(df):
    """
    Enrich earthquake data with additional features
    
    Args:
        df (pandas.DataFrame): Earthquake dataframe
    
    Returns:
        pandas.DataFrame: Enriched dataframe
    """
    if df.empty:
        return df
    
    # Copy to avoid modifying the original
    enriched_df = df.copy()
    
    # Add depth category
    if 'depth' in enriched_df.columns:
        conditions = [
            (enriched_df['depth'] < 70),
            (enriched_df['depth'] >= 70) & (enriched_df['depth'] < 300),
            (enriched_df['depth'] >= 300)
        ]
        categories = ['Shallow', 'Intermediate', 'Deep']
        enriched_df['depth_category'] = np.select(conditions, categories, default='Unknown')
    
    # Add magnitude category
    if 'magnitude' in enriched_df.columns:
        conditions = [
            (enriched_df['magnitude'] < 4.0),
            (enriched_df['magnitude'] >= 4.0) & (enriched_df['magnitude'] < 5.0),
            (enriched_df['magnitude'] >= 5.0) & (enriched_df['magnitude'] < 6.0),
            (enriched_df['magnitude'] >= 6.0) & (enriched_df['magnitude'] < 7.0),
            (enriched_df['magnitude'] >= 7.0)
        ]
        categories = ['Minor', 'Light', 'Moderate', 'Strong', 'Major']
        enriched_df['magnitude_category'] = np.select(conditions, categories, default='Unknown')
    
    # Add tsunami risk indicator
    if 'tsunami' in enriched_df.columns:
        enriched_df['tsunami_risk'] = enriched_df['tsunami'] > 0
    elif 'magnitude' in enriched_df.columns:
        # Estimate tsunami risk based on magnitude (simplified)
        enriched_df['tsunami_risk'] = enriched_df['magnitude'] >= 7.0
    
    return enriched_df

def enrich_hurricane_data(df):
    """
    Enrich hurricane data with additional features
    
    Args:
        df (pandas.DataFrame): Hurricane dataframe
    
    Returns:
        pandas.DataFrame: Enriched dataframe
    """
    if df.empty:
        return df
    
    # Copy to avoid modifying the original
    enriched_df = df.copy()
    
    # Add Saffir-Simpson hurricane category if not already present
    if 'category' not in enriched_df.columns and 'wind_speed' in enriched_df.columns:
        conditions = [
            (enriched_df['wind_speed'] < 39),
            (enriched_df['wind_speed'] >= 39) & (enriched_df['wind_speed'] < 74),
            (enriched_df['wind_speed'] >= 74) & (enriched_df['wind_speed'] < 96),
            (enriched_df['wind_speed'] >= 96) & (enriched_df['wind_speed'] < 111),
            (enriched_df['wind_speed'] >= 111) & (enriched_df['wind_speed'] < 130),
            (enriched_df['wind_speed'] >= 130) & (enriched_df['wind_speed'] < 157),
            (enriched_df['wind_speed'] >= 157)
        ]
        categories = ['Tropical Depression', 'Tropical Storm', 'Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5']
        enriched_df['category'] = np.select(conditions, categories, default='Unknown')
    
    # Add risk level based on category or wind speed
    if 'category' in enriched_df.columns and enriched_df['category'].dtype == 'int64':
        enriched_df['risk_level'] = enriched_df['category'].apply(
            lambda x: 'Low' if x < 3 else ('High' if x >= 4 else 'Medium')
        )
    elif 'wind_speed' in enriched_df.columns:
        conditions = [
            (enriched_df['wind_speed'] < 74),
            (enriched_df['wind_speed'] >= 74) & (enriched_df['wind_speed'] < 111),
            (enriched_df['wind_speed'] >= 111)
        ]
        risk_levels = ['Low', 'Medium', 'High']
        enriched_df['risk_level'] = np.select(conditions, risk_levels, default='Unknown')
    
    return enriched_df

def enrich_flood_data(df):
    """
    Enrich flood data with additional features
    
    Args:
        df (pandas.DataFrame): Flood dataframe
    
    Returns:
        pandas.DataFrame: Enriched dataframe
    """
    if df.empty:
        return df
    
    # Copy to avoid modifying the original
    enriched_df = df.copy()
    
    # Add severity category if not already present
    if 'severity_category' not in enriched_df.columns and 'severity_value' in enriched_df.columns:
        conditions = [
            (enriched_df['severity_value'] < 1.0),
            (enriched_df['severity_value'] >= 1.0) & (enriched_df['severity_value'] < 2.0),
            (enriched_df['severity_value'] >= 2.0)
        ]
        categories = ['Minor', 'Moderate', 'Severe']
        enriched_df['severity_category'] = np.select(conditions, categories, default='Unknown')
    
    # Add impact level based on population affected
    if 'population_affected' in enriched_df.columns:
        conditions = [
            (enriched_df['population_affected'] < 1000),
            (enriched_df['population_affected'] >= 1000) & (enriched_df['population_affected'] < 10000),
            (enriched_df['population_affected'] >= 10000)
        ]
        impact_levels = ['Low', 'Medium', 'High']
        enriched_df['impact_level'] = np.select(conditions, impact_levels, default='Unknown')
    
    # Add area impact level
    if 'area_affected_km2' in enriched_df.columns:
        conditions = [
            (enriched_df['area_affected_km2'] < 500),
            (enriched_df['area_affected_km2'] >= 500) & (enriched_df['area_affected_km2'] < 2000),
            (enriched_df['area_affected_km2'] >= 2000)
        ]
        area_impacts = ['Localized', 'Regional', 'Widespread']
        enriched_df['area_impact'] = np.select(conditions, area_impacts, default='Unknown')
    
    return enriched_df
