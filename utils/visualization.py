import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_risk_map(data, disaster_type, is_prediction=False):
    """
    Create an interactive map showing disaster locations and risk areas

    Args:
        data (pandas.DataFrame): DataFrame containing location and intensity data
        disaster_type (str): Type of disaster (Earthquake, Hurricane, Flood)
        is_prediction (bool): Whether this is a prediction map

    Returns:
        folium.Map: Interactive folium map
    """
    # Create a base map
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="cartodbpositron")
    
    # Add title
    title = f"{'Predicted' if is_prediction else 'Current'} {disaster_type} Activity"
    title_html = f'''
        <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Handle different disaster types
    if disaster_type == "Earthquake":
        _add_earthquake_layers(m, data, is_prediction, disaster_type)
    elif disaster_type == "Hurricane":
        _add_hurricane_layers(m, data, is_prediction, disaster_type)
    elif disaster_type == "Flood":
        _add_flood_layers(m, data, is_prediction, disaster_type)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def _add_earthquake_layers(m, data, is_prediction, disaster_type="Earthquake"):
    """
    Add earthquake-specific layers to the map

    Args:
        m (folium.Map): Map to add layers to
        data (pandas.DataFrame): Earthquake data
        is_prediction (bool): Whether this is a prediction map
        disaster_type (str): Type of disaster (default: "Earthquake")
    """
    # Create marker clusters for earthquakes
    marker_cluster = MarkerCluster(name="Earthquakes").add_to(m)
    
    # Extract necessary columns
    if not data.empty:
        # Ensure required columns exist
        required_cols = ['latitude', 'longitude']
        intensity_col = 'predicted_intensity' if is_prediction else 'magnitude'
        
        if all(col in data.columns for col in required_cols) and intensity_col in data.columns:
            # Create color mapping based on intensity
            def get_color(intensity):
                if intensity < 4.0:
                    return 'green'
                elif intensity < 5.0:
                    return 'yellow'
                elif intensity < 6.0:
                    return 'orange'
                else:
                    return 'red'
            
            # Add individual markers
            for idx, row in data.iterrows():
                intensity = row[intensity_col]
                color = get_color(intensity)
                
                # Determine popup content
                if is_prediction:
                    confidence = row.get('confidence', 'N/A')
                    if isinstance(confidence, (float, int)):
                        confidence_display = f"{confidence:.2f}"
                    else:
                        confidence_display = str(confidence)
                    
                    popup_html = f"""
                    <b>Predicted Magnitude:</b> {intensity:.1f}<br>
                    <b>Confidence:</b> {confidence_display}<br>
                    <b>Location:</b> {row.get('location', 'Unknown')}<br>
                    """
                else:
                    popup_html = f"""
                    <b>Magnitude:</b> {intensity:.1f}<br>
                    <b>Depth:</b> {row.get('depth', 'N/A')} km<br>
                    <b>Location:</b> {row.get('place', row.get('location', 'Unknown'))}<br>
                    <b>Time:</b> {row.get('time', 'N/A')}<br>
                    """
                
                # Create marker
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5 + intensity * 1.5,  # Size based on intensity
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{intensity:.1f} {'Predicted' if is_prediction else ''} {disaster_type}"
                ).add_to(marker_cluster)
            
            # Add heatmap layer for density visualization
            heat_data = data[['latitude', 'longitude', intensity_col]].values.tolist()
            HeatMap(
                heat_data,
                name="Heat Map",
                radius=15,
                min_opacity=0.3,
                gradient={'0.4': 'blue', '0.65': 'lime', '0.8': 'orange', '1.0': 'red'}
            ).add_to(m)
            
            # Add tectonic plate boundaries for context with earthquakes
            folium.GeoJson(
                "https://raw.githubusercontent.com/fraxen/tectonicplates/master/GeoJSON/PB2002_boundaries.json",
                name="Tectonic Plates",
                style_function=lambda x: {
                    'color': '#606060',
                    'weight': 2,
                    'opacity': 0.5
                }
            ).add_to(m)

def _add_hurricane_layers(m, data, is_prediction, disaster_type="Hurricane"):
    """
    Add hurricane-specific layers to the map

    Args:
        m (folium.Map): Map to add layers to
        data (pandas.DataFrame): Hurricane data
        is_prediction (bool): Whether this is a prediction map
        disaster_type (str): Type of disaster (default: "Hurricane")
    """
    if not data.empty:
        # Ensure required columns exist
        required_cols = ['latitude', 'longitude', 'name']
        intensity_col = 'predicted_intensity' if is_prediction else 'wind_speed'
        
        if all(col in data.columns for col in required_cols) and intensity_col in data.columns:
            # Group data by storm name
            storm_names = data['name'].unique()
            
            # Create a layer for each storm
            for storm_name in storm_names:
                storm_data = data[data['name'] == storm_name]
                
                # Sort by position number or time if available
                if 'position_number' in storm_data.columns:
                    storm_data = storm_data.sort_values('position_number')
                elif 'time' in storm_data.columns:
                    storm_data = storm_data.sort_values('time')
                
                # Create a color based on hurricane category or intensity
                def get_hurricane_color(intensity):
                    if intensity < 75:  # Tropical storm
                        return '#a6cee3'
                    elif intensity < 96:  # Category 1
                        return '#1f78b4'
                    elif intensity < 111:  # Category 2
                        return '#b2df8a'
                    elif intensity < 130:  # Category 3
                        return '#33a02c'
                    elif intensity < 157:  # Category 4
                        return '#fb9a99'
                    else:  # Category 5
                        return '#e31a1c'
                
                # Create path for the storm track
                track_points = storm_data[['latitude', 'longitude']].values.tolist()
                if len(track_points) > 1:
                    # Determine color based on max intensity
                    max_intensity = storm_data[intensity_col].max()
                    color = get_hurricane_color(max_intensity)
                    
                    # Draw the hurricane path
                    folium.PolyLine(
                        track_points,
                        color=color,
                        weight=3,
                        opacity=0.8,
                        tooltip=f"{storm_name} Path"
                    ).add_to(m)
                
                # Add markers for each position
                for idx, row in storm_data.iterrows():
                    intensity = row[intensity_col]
                    color = get_hurricane_color(intensity)
                    
                    # Determine marker size based on intensity
                    radius = 5 + (intensity / 20)
                    
                    # Create popup content
                    if is_prediction:
                        time_str = row.get('time', 'Unknown time')
                        if isinstance(time_str, datetime):
                            time_str = time_str.strftime('%Y-%m-%d %H:%M')
                            
                        confidence = row.get('confidence', 'N/A')
                        if isinstance(confidence, (float, int)):
                            confidence_display = f"{confidence:.2f}"
                        else:
                            confidence_display = str(confidence)
                            
                        popup_html = f"""
                        <b>Storm:</b> {storm_name}<br>
                        <b>Predicted Wind Speed:</b> {intensity:.1f} mph<br>
                        <b>Confidence:</b> {confidence_display}<br>
                        <b>Predicted Time:</b> {time_str}<br>
                        <b>Position:</b> {row.get('position_number', 'N/A')}<br>
                        """
                    else:
                        time_str = row.get('time', 'Unknown time')
                        if isinstance(time_str, datetime):
                            time_str = time_str.strftime('%Y-%m-%d %H:%M')
                            
                        popup_html = f"""
                        <b>Storm:</b> {storm_name}<br>
                        <b>Wind Speed:</b> {intensity:.1f} mph<br>
                        <b>Category:</b> {row.get('category', 'N/A')}<br>
                        <b>Pressure:</b> {row.get('pressure', 'N/A')} mb<br>
                        <b>Time:</b> {time_str}<br>
                        <b>Position:</b> {row.get('position_number', 'N/A')}<br>
                        """
                    
                    # Add a marker for each position
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=f"{storm_name} - {intensity:.1f} mph"
                    ).add_to(m)

def _add_flood_layers(m, data, is_prediction, disaster_type="Flood"):
    """
    Add flood-specific layers to the map

    Args:
        m (folium.Map): Map to add layers to
        data (pandas.DataFrame): Flood data
        is_prediction (bool): Whether this is a prediction map
        disaster_type (str): Type of disaster (default: "Flood")
    """
    if not data.empty:
        # Ensure required columns exist
        required_cols = ['latitude', 'longitude']
        intensity_col = 'predicted_intensity' if is_prediction else 'severity_value'
        
        if all(col in data.columns for col in required_cols) and intensity_col in data.columns:
            # Create a feature group for flood markers
            flood_layer = folium.FeatureGroup(name="Floods")
            
            # Add markers for flood areas
            for idx, row in data.iterrows():
                intensity = row[intensity_col]
                
                # Determine color based on severity
                def get_flood_color(intensity):
                    if intensity < 1.0:
                        return '#a6cee3'  # Light blue - Minor
                    elif intensity < 2.0:
                        return '#1f78b4'  # Medium blue - Moderate
                    else:
                        return '#08306b'  # Dark blue - Severe
                
                color = get_flood_color(intensity)
                
                # Determine radius based on affected area if available
                radius = 10  # Default radius
                if 'area_affected_km2' in row:
                    # Scale radius based on area (square root for visual proportionality)
                    radius = 5 + (np.sqrt(row['area_affected_km2']) / 10)
                    radius = min(50, radius)  # Cap at 50
                
                # Create popup content
                if is_prediction:
                    confidence = row.get('confidence', 'N/A')
                    if isinstance(confidence, (float, int)):
                        confidence_display = f"{confidence:.2f}"
                    else:
                        confidence_display = str(confidence)
                        
                    popup_html = f"""
                    <b>Area:</b> {row.get('name', 'Unknown')}<br>
                    <b>Predicted Severity:</b> {intensity:.1f}/3<br>
                    <b>Confidence:</b> {confidence_display}<br>
                    <b>Time:</b> {row.get('time', 'N/A')}<br>
                    """
                else:
                    popup_html = f"""
                    <b>Area:</b> {row.get('name', 'Unknown')}<br>
                    <b>Severity:</b> {intensity:.1f}/3<br>
                    <b>Area Affected:</b> {row.get('area_affected_km2', 'N/A')} km²<br>
                    <b>Population Affected:</b> {row.get('population_affected', 'N/A'):,}<br>
                    <b>Time:</b> {row.get('time', 'N/A')}<br>
                    """
                
                # Add a marker for each flood area
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Flood Severity: {intensity:.1f}/3"
                ).add_to(flood_layer)
            
            # Add the flood layer to the map
            flood_layer.add_to(m)
            
            # Add heatmap layer if we have enough data points
            if len(data) >= 5:
                heat_data = data[['latitude', 'longitude', intensity_col]].values.tolist()
                HeatMap(
                    heat_data,
                    name="Flood Intensity Heat Map",
                    radius=20,
                    min_opacity=0.4,
                    gradient={'0.4': '#89cff0', '0.65': '#4682b4', '0.8': '#0000cd', '1.0': '#00008b'}
                ).add_to(m)

def create_historical_chart(data, disaster_type):
    """
    Create a chart visualizing historical disaster data

    Args:
        data (pandas.DataFrame): Historical disaster data with date field
        disaster_type (str): Type of disaster (Earthquake, Hurricane, Flood)

    Returns:
        plotly.graph_objects.Figure: Interactive time series chart
    """
    if data.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No historical data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Determine y-axis column based on disaster type
    if disaster_type == "Earthquake":
        y_column = 'magnitude'
        title = 'Historical Earthquake Magnitudes'
        y_title = 'Magnitude'
    elif disaster_type == "Hurricane":
        if 'wind_speed' in data.columns:
            y_column = 'wind_speed'
            title = 'Historical Hurricane Wind Speeds'
            y_title = 'Wind Speed (mph)'
        else:
            y_column = 'active_count'
            title = 'Historical Hurricane Activity'
            y_title = 'Active Storms'
    elif disaster_type == "Flood":
        if 'severity_value' in data.columns:
            y_column = 'severity_value'
            title = 'Historical Flood Severity'
            y_title = 'Severity (0-3 scale)'
        elif 'area_affected' in data.columns:
            y_column = 'area_affected'
            title = 'Historical Flood Affected Areas'
            y_title = 'Area Affected (km²)'
        else:
            y_column = 'casualties'
            title = 'Historical Flood Casualties'
            y_title = 'Number of Casualties'
    else:
        y_column = data.columns[1]  # Default to second column
        title = 'Historical Data'
        y_title = y_column.replace('_', ' ').title()
    
    # Create time series chart
    fig = px.line(
        data, 
        x='date', 
        y=y_column,
        title=title,
        labels={'date': 'Date', y_column: y_title}
    )
    
    # Add scatter points for individual events
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data[y_column],
            mode='markers',
            name='Events',
            marker=dict(
                size=8,
                opacity=0.6,
                line=dict(width=1, color='DarkSlateGrey')
            )
        )
    )
    
    # Add moving average for trend visualization
    window_size = min(10, max(3, len(data) // 10))
    data_sorted = data.sort_values('date')
    data_sorted['moving_avg'] = data_sorted[y_column].rolling(window=window_size, center=True).mean()
    
    fig.add_trace(
        go.Scatter(
            x=data_sorted['date'],
            y=data_sorted['moving_avg'],
            mode='lines',
            name=f'{window_size}-point Moving Average',
            line=dict(color='red', width=2, dash='dot')
        )
    )
    
    # Enhance the layout
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='LightGray',
            title_font=dict(size=14),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='LightGray',
            title_font=dict(size=14),
            tickfont=dict(size=12)
        )
    )
    
    # Add annotations for major events if available
    if 'region' in data.columns:
        # Find top 3 most significant events
        if disaster_type == "Earthquake":
            # For earthquakes, significance is based on magnitude
            major_events = data.nlargest(3, y_column)
        elif disaster_type == "Hurricane":
            # For hurricanes, significance is based on wind speed or damage
            if 'damage_millions' in data.columns:
                major_events = data.nlargest(3, 'damage_millions')
            else:
                major_events = data.nlargest(3, y_column)
        elif disaster_type == "Flood":
            # For floods, significance is based on area or casualties
            if 'casualties' in data.columns:
                major_events = data.nlargest(3, 'casualties')
            else:
                major_events = data.nlargest(3, y_column)
        else:
            major_events = data.nlargest(3, y_column)
        
        # Add annotations for major events
        for i, event in major_events.iterrows():
            fig.add_annotation(
                x=event['date'],
                y=event[y_column],
                text=f"{event['region']}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black",
                ax=-30,
                ay=-30
            )
    
    return fig

def create_forecast_chart(predictions, disaster_type):
    """
    Create a chart visualizing forecast data

    Args:
        predictions (pandas.DataFrame): Prediction data including forecasts
        disaster_type (str): Type of disaster (Earthquake, Hurricane, Flood)

    Returns:
        plotly.graph_objects.Figure: Interactive forecast chart
    """
    if predictions.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No prediction data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Determine if there are historical and prediction values
    has_predictions = 'is_prediction' in predictions.columns
    
    # Determine y-axis column based on disaster type
    if 'predicted_intensity' in predictions.columns:
        y_column = 'predicted_intensity'
    elif disaster_type == "Earthquake":
        y_column = 'magnitude'
    elif disaster_type == "Hurricane":
        y_column = 'wind_speed' if 'wind_speed' in predictions.columns else 'category'
    elif disaster_type == "Flood":
        y_column = 'severity_value' if 'severity_value' in predictions.columns else 'area_affected_km2'
    else:
        # Default to first numeric column that's not lat/long
        for col in predictions.columns:
            if col not in ['latitude', 'longitude', 'is_prediction', 'time', 'date'] and pd.api.types.is_numeric_dtype(predictions[col]):
                y_column = col
                break
    
    # Create title based on disaster type
    title_map = {
        "Earthquake": "Earthquake Magnitude Forecast",
        "Hurricane": "Hurricane Intensity Forecast",
        "Flood": "Flood Severity Forecast"
    }
    title = title_map.get(disaster_type, "Disaster Forecast")
    
    # Create empty figure
    fig = go.Figure()
    
    # Add historical data if available
    if has_predictions:
        historical = predictions[predictions['is_prediction'] == False]
        future = predictions[predictions['is_prediction'] == True]
        
        if not historical.empty:
            # Sort by time if available
            if 'time' in historical.columns:
                historical = historical.sort_values('time')
                x_column = 'time'
            else:
                x_column = historical.index
            
            fig.add_trace(
                go.Scatter(
                    x=historical[x_column],
                    y=historical[y_column],
                    mode='markers+lines',
                    name='Historical',
                    marker=dict(color='blue', size=8)
                )
            )
        
        if not future.empty:
            # Sort by time if available
            if 'time' in future.columns:
                future = future.sort_values('time')
                x_column = 'time'
            else:
                x_column = future.index
            
            # Add prediction line
            fig.add_trace(
                go.Scatter(
                    x=future[x_column],
                    y=future[y_column],
                    mode='markers+lines',
                    name='Forecast',
                    marker=dict(color='red', size=10),
                    line=dict(color='red', dash='dash')
                )
            )
            
            # Add confidence intervals if available
            if 'confidence' in future.columns:
                # Calculate confidence intervals
                confidence_values = future['confidence'].values
                y_values = future[y_column].values
                
                upper_bound = y_values + (1 - confidence_values) * y_values * 0.5
                lower_bound = y_values - (1 - confidence_values) * y_values * 0.5
                
                # Add confidence interval
                fig.add_trace(
                    go.Scatter(
                        x=future[x_column].tolist() + future[x_column].tolist()[::-1],
                        y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,0,0,0)'),
                        name='Confidence Interval',
                        showlegend=True
                    )
                )
    else:
        # If no prediction indicator, assume all are predictions
        # Sort by time if available
        if 'time' in predictions.columns:
            predictions = predictions.sort_values('time')
            x_column = 'time'
        else:
            x_column = predictions.index
        
        fig.add_trace(
            go.Scatter(
                x=predictions[x_column],
                y=predictions[y_column],
                mode='markers+lines',
                name='Forecast',
                marker=dict(color='red', size=10),
                line=dict(color='red', dash='dash')
            )
        )
    
    # Add labels and title
    y_title = y_column.replace('_', ' ').replace('predicted ', '').title()
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_title,
        legend_title="Data Type",
        hovermode="x unified",
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='LightGray',
            title_font=dict(size=14),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='LightGray',
            title_font=dict(size=14),
            tickfont=dict(size=12)
        )
    )
    
    return fig
