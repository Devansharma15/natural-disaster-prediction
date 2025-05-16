import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
import datetime
from data_fetcher import (
    fetch_earthquake_data, 
    fetch_hurricane_data, 
    fetch_flood_data,
    fetch_historical_disasters
)
from models.disaster_predictor import DisasterPredictor
from models.time_series_analyzer import TimeSeriesAnalyzer
from utils.visualization import (
    create_risk_map, 
    create_historical_chart, 
    create_forecast_chart
)
from utils.data_processor import (
    preprocess_earthquake_data, 
    preprocess_hurricane_data, 
    preprocess_flood_data
)

# Configure the page
st.set_page_config(
    page_title="Natural Disaster Prediction System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state if not already done
if 'disaster_type' not in st.session_state:
    st.session_state['disaster_type'] = 'Earthquake'
if 'time_range' not in st.session_state:
    st.session_state['time_range'] = '7 days'
if 'region' not in st.session_state:
    st.session_state['region'] = 'Global'
if 'last_updated' not in st.session_state:
    st.session_state['last_updated'] = None

# Header
st.title("🌊 Real-Time Natural Disaster Prediction System")

# Sidebar
st.sidebar.header("Settings")

# Disaster type selection
disaster_type = st.sidebar.selectbox(
    "Disaster Type",
    ["Earthquake", "Hurricane", "Flood"],
    key="disaster_type_select"
)

# Time range selection
time_range = st.sidebar.selectbox(
    "Time Range",
    ["1 day", "7 days", "30 days", "90 days"],
    index=1,
    key="time_range_select"
)

# Region selection
region = st.sidebar.selectbox(
    "Region",
    ["Global", "North America", "South America", "Europe", "Asia", "Africa", "Oceania"],
    key="region_select"
)

# Update session state when user changes selections
if disaster_type != st.session_state['disaster_type'] or \
   time_range != st.session_state['time_range'] or \
   region != st.session_state['region']:
    st.session_state['disaster_type'] = disaster_type
    st.session_state['time_range'] = time_range
    st.session_state['region'] = region

# Refresh button
if st.sidebar.button("Refresh Data"):
    st.session_state['last_updated'] = datetime.datetime.now()
    st.rerun()

# Show last updated time
if st.session_state['last_updated']:
    st.sidebar.write(f"Last updated: {st.session_state['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")

# Model settings (collapsible)
with st.sidebar.expander("Model Settings"):
    model_type = st.selectbox(
        "Prediction Model",
        ["Random Forest", "LSTM"],
        index=0
    )
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05
    )

# About section in sidebar
with st.sidebar.expander("About"):
    st.write("""
    This application provides real-time natural disaster predictions and visualizations using 
    machine learning models and data from official sources like USGS, NOAA, and NASA.
    
    Data sources:
    - Earthquake: USGS Earthquake Catalog
    - Hurricane: NOAA National Hurricane Center
    - Flood: NASA Global Flood Monitoring
    """)

# Links to model testing interface
st.sidebar.markdown("---")
st.sidebar.markdown("[Open Model Testing Interface](http://localhost:5000/model_testing)")
st.sidebar.markdown("[GitHub Repository](https://github.com/example/natural-disaster-prediction)")

# Main content area
st.markdown("## Current Disaster Activity")

# Loading indicator
with st.spinner("Fetching latest disaster data..."):
    # Fetch real-time data based on user selection
    if disaster_type == "Earthquake":
        raw_data = fetch_earthquake_data(time_range)
        data = preprocess_earthquake_data(raw_data, region)
    elif disaster_type == "Hurricane":
        raw_data = fetch_hurricane_data(time_range)
        data = preprocess_hurricane_data(raw_data, region)
    elif disaster_type == "Flood":
        raw_data = fetch_flood_data(time_range)
        data = preprocess_flood_data(raw_data, region)

# Display data metrics
col1, col2, col3 = st.columns(3)

with col1:
    if disaster_type == "Earthquake":
        st.metric("Active Seismic Zones", len(data['active_zones']) if 'active_zones' in data else 0)
    elif disaster_type == "Hurricane":
        st.metric("Active Storms", len(data['active_storms']) if 'active_storms' in data else 0)
    elif disaster_type == "Flood":
        st.metric("Affected Areas", len(data['affected_areas']) if 'affected_areas' in data else 0)

with col2:
    if disaster_type == "Earthquake":
        st.metric("Highest Magnitude", data['max_magnitude'] if 'max_magnitude' in data else "N/A")
    elif disaster_type == "Hurricane":
        st.metric("Highest Category", data['max_category'] if 'max_category' in data else "N/A")
    elif disaster_type == "Flood":
        st.metric("Highest Severity", data['max_severity'] if 'max_severity' in data else "N/A")

with col3:
    if disaster_type == "Earthquake":
        st.metric("Total Events", data['total_events'] if 'total_events' in data else 0)
    elif disaster_type == "Hurricane":
        st.metric("Wind Speed (Max)", f"{data['max_wind_speed']} mph" if 'max_wind_speed' in data else "N/A")
    elif disaster_type == "Flood":
        st.metric("Land Area Affected", f"{data['total_area']:,} km²" if 'total_area' in data else "N/A")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Map View", "Historical Data", "Predictions"])

with tab1:
    st.markdown("### Global Distribution and Risk Areas")
    
    # Create interactive map
    if 'data_for_map' in data:
        risk_map = create_risk_map(data['data_for_map'], disaster_type)
        folium_static(risk_map, width=1000, height=500)
    else:
        st.warning("No map data available for the selected criteria.")
    
    # Show data table
    with st.expander("View Data Table"):
        if 'dataframe' in data:
            st.dataframe(data['dataframe'])
        else:
            st.warning("No detailed data available for the selected criteria.")

with tab2:
    st.markdown("### Historical Disaster Analysis")
    
    # Historical data visualization
    historical_data = fetch_historical_disasters(disaster_type, time_range, region)
    
    if not historical_data.empty:
        # Create time series chart
        fig = create_historical_chart(historical_data, disaster_type)
        st.plotly_chart(fig, use_container_width=True)
        
        # Analysis metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Trend Analysis")
            # Use time series analyzer to get trend metrics
            ts_analyzer = TimeSeriesAnalyzer()
            trend_metrics = ts_analyzer.analyze_trends(historical_data, disaster_type)
            
            if trend_metrics:
                for metric, value in trend_metrics.items():
                    st.metric(metric, value)
            else:
                st.warning("Insufficient data for trend analysis.")
        
        with col2:
            st.markdown("#### Seasonal Patterns")
            # Show seasonal patterns if available
            if disaster_type in ["Hurricane", "Flood"]:
                seasonal_fig = ts_analyzer.plot_seasonality(historical_data, disaster_type)
                st.plotly_chart(seasonal_fig, use_container_width=True)
            else:
                st.info("Seasonal analysis not applicable for earthquakes.")
    else:
        st.warning("No historical data available for the selected criteria.")

with tab3:
    st.markdown("### Disaster Predictions")
    
    # Initialize disaster predictor with selected model
    predictor = DisasterPredictor(model_type=model_type)
    
    # Make predictions
    with st.spinner("Generating predictions..."):
        if 'dataframe' in data:
            predictions = predictor.predict(data['dataframe'], disaster_type)
            
            if predictions is not None:
                # Show prediction map
                pred_map = create_risk_map(predictions, disaster_type, is_prediction=True)
                st.markdown("#### Predicted Risk Areas (Next 7 Days)")
                folium_static(pred_map, width=1000, height=500)
                
                # Show forecast chart
                st.markdown("#### Forecast Intensity")
                forecast_fig = create_forecast_chart(predictions, disaster_type)
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Confidence metrics
                st.markdown("#### Prediction Confidence")
                confidence_data = predictor.get_confidence_metrics()
                
                conf_col1, conf_col2, conf_col3 = st.columns(3)
                with conf_col1:
                    st.metric("Mean Confidence", f"{confidence_data['mean_confidence']:.2f}")
                with conf_col2:
                    st.metric("Model Accuracy", f"{confidence_data['model_accuracy']:.2f}")
                with conf_col3:
                    st.metric("Data Quality", f"{confidence_data['data_quality']:.2f}")
                
                # Alert information if confidence is high enough
                if confidence_data['mean_confidence'] > confidence_threshold:
                    st.warning("⚠️ High-confidence prediction of significant activity detected. Monitor situation closely.")
            else:
                st.info("Insufficient data for reliable predictions. Try changing the filters.")
        else:
            st.warning("No data available for predictions. Try refreshing or changing your filters.")

# Add resources and help
st.markdown("---")
st.markdown("### Resources & Help")
resource_col1, resource_col2 = st.columns(2)

with resource_col1:
    st.markdown("""
    #### Emergency Contacts
    - **Global Disaster Alert**: [gdacs.org](https://www.gdacs.org/)
    - **Red Cross**: [redcross.org](https://www.redcross.org/)
    - **FEMA**: [fema.gov](https://www.fema.gov/)
    """)

with resource_col2:
    st.markdown("""
    #### Data Sources
    - **Earthquake Data**: [USGS](https://earthquake.usgs.gov/earthquakes/feed/)
    - **Hurricane Data**: [NOAA](https://www.nhc.noaa.gov/data/)
    - **Flood Data**: [Global Flood Monitoring](https://global.floods.rti.org/)
    """)

# Footer
st.markdown("---")
st.caption("© 2023 Natural Disaster Prediction System | Data refreshes automatically every 6 hours")
