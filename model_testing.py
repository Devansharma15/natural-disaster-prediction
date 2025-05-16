import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import os
# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.disaster_predictor import DisasterPredictor
from models.time_series_analyzer import TimeSeriesAnalyzer
from utils.data_processor import (
    preprocess_earthquake_data, 
    preprocess_hurricane_data, 
    preprocess_flood_data
)
from data_fetcher import (
    fetch_earthquake_data, 
    fetch_sample_data
)
import streamlit as st

def load_sample_data(disaster_type):
    """
    Load sample data for model testing based on disaster type
    """
    return fetch_sample_data(disaster_type)

def run_prediction(disaster_type, model_type, time_range, lat_range, long_range, magnitude_range=None):
    """
    Run a prediction based on user input parameters
    """
    if disaster_type == "Earthquake":
        # Get real data as a starting point
        base_data = fetch_earthquake_data(time_range)
        data = preprocess_earthquake_data(base_data, "Global")
        
        # Filter the data based on user parameters
        if 'dataframe' in data and not data['dataframe'].empty:
            filtered_df = data['dataframe']
            filtered_df = filtered_df[
                (filtered_df['latitude'] >= lat_range[0]) & 
                (filtered_df['latitude'] <= lat_range[1]) &
                (filtered_df['longitude'] >= long_range[0]) & 
                (filtered_df['longitude'] <= long_range[1])
            ]
            
            if magnitude_range:
                filtered_df = filtered_df[
                    (filtered_df['magnitude'] >= magnitude_range[0]) & 
                    (filtered_df['magnitude'] <= magnitude_range[1])
                ]
                
            # Create predictor and make predictions
            predictor = DisasterPredictor(model_type=model_type)
            predictions = predictor.predict(filtered_df, disaster_type)
            
            # Get confidence metrics
            conf_metrics = predictor.get_confidence_metrics()
            
            # Create visualization for predictions
            if not predictions.empty:
                fig = px.scatter_mapbox(
                    predictions, 
                    lat="latitude", 
                    lon="longitude", 
                    color="predicted_intensity",
                    size="predicted_intensity",
                    hover_name="location", 
                    hover_data=["predicted_intensity", "confidence"],
                    mapbox_style="carto-positron",
                    title="Earthquake Prediction Results",
                    zoom=2,
                    color_continuous_scale="Viridis"
                )
                
                # Create a data summary table
                summary_df = pd.DataFrame({
                    "Metric": ["Number of predictions", "Mean predicted intensity", "Mean confidence",
                              "Max predicted intensity", "Min predicted intensity"],
                    "Value": [
                        len(predictions),
                        round(predictions["predicted_intensity"].mean(), 2),
                        round(predictions["confidence"].mean(), 2),
                        round(predictions["predicted_intensity"].max(), 2),
                        round(predictions["predicted_intensity"].min(), 2)
                    ]
                })
                
                return fig, summary_df, conf_metrics
            else:
                return None, pd.DataFrame({"Error": ["No predictions generated"]}), None
        else:
            return None, pd.DataFrame({"Error": ["No data available for the selected criteria"]}), None
    
    elif disaster_type == "Hurricane":
        # Similar implementation for hurricane data
        # For brevity, I'm keeping this as a placeholder since the logic would be similar
        return None, pd.DataFrame({"Status": ["Hurricane prediction not implemented in this demo"]}), None
        
    elif disaster_type == "Flood":
        # Similar implementation for flood data
        # For brevity, I'm keeping this as a placeholder since the logic would be similar
        return None, pd.DataFrame({"Status": ["Flood prediction not implemented in this demo"]}), None
    
    return None, pd.DataFrame({"Error": ["Invalid disaster type"]}), None

def evaluate_model(disaster_type, model_type, test_data=None):
    """
    Evaluate a model's performance on historical data
    """
    if test_data is None:
        test_data = load_sample_data(disaster_type)
    
    # Initialize the predictor
    predictor = DisasterPredictor(model_type=model_type)
    
    # Do a train/test split if we have enough data
    if len(test_data) > 100:
        train_size = int(0.8 * len(test_data))
        train_data = test_data[:train_size]
        test_data = test_data[train_size:]
    else:
        train_data = test_data
    
    # Train the model
    predictor.train(train_data, disaster_type)
    
    # Evaluate on test data
    eval_metrics = predictor.evaluate(test_data, disaster_type)
    
    # Create performance visualization
    fig = px.line(
        x=range(len(eval_metrics['actual_vs_predicted'])),
        y=[eval_metrics['actual_vs_predicted']['actual'], eval_metrics['actual_vs_predicted']['predicted']],
        labels={'x': 'Sample Index', 'y': 'Intensity/Magnitude'},
        title=f"{disaster_type} Prediction Model Evaluation",
        color_discrete_sequence=["blue", "red"]
    )
    fig.update_layout(legend_title_text='Data Type')
    
    # Create metrics summary
    metrics_df = pd.DataFrame({
        "Metric": list(eval_metrics['metrics'].keys()),
        "Value": list(eval_metrics['metrics'].values())
    })
    
    return fig, metrics_df, eval_metrics

def create_interface():
    """
    Create the Gradio interface for model testing
    """
    with gr.Blocks(title="Natural Disaster Prediction - Model Testing") as interface:
        gr.Markdown("# Natural Disaster Prediction Model Testing")
        gr.Markdown("This interface allows you to test and evaluate prediction models for different types of natural disasters.")
        
        with gr.Tab("Run Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    disaster_type = gr.Dropdown(
                        ["Earthquake", "Hurricane", "Flood"], 
                        label="Disaster Type",
                        value="Earthquake"
                    )
                    model_type = gr.Dropdown(
                        ["Random Forest", "LSTM"], 
                        label="Model Type",
                        value="Random Forest"
                    )
                    time_range = gr.Dropdown(
                        ["1 day", "7 days", "30 days", "90 days"], 
                        label="Time Range",
                        value="7 days"
                    )
                
                with gr.Column(scale=1):
                    lat_range = gr.Slider(
                        minimum=-90, 
                        maximum=90, 
                        value=(-90, 90), 
                        label="Latitude Range",
                        step=1
                    )
                    long_range = gr.Slider(
                        minimum=-180, 
                        maximum=180, 
                        value=(-180, 180), 
                        label="Longitude Range",
                        step=1
                    )
                    magnitude_slider = gr.Slider(
                        minimum=0, 
                        maximum=10, 
                        value=(4, 10), 
                        label="Magnitude Range (Earthquake only)",
                        step=0.1
                    )
            
            run_button = gr.Button("Run Prediction")
            
            with gr.Row():
                with gr.Column(scale=2):
                    prediction_plot = gr.Plot(label="Prediction Visualization")
                with gr.Column(scale=1):
                    prediction_data = gr.Dataframe(label="Prediction Summary")
                    confidence_metrics = gr.JSON(label="Confidence Metrics")
            
            run_button.click(
                run_prediction,
                [disaster_type, model_type, time_range, lat_range, long_range, magnitude_slider],
                [prediction_plot, prediction_data, confidence_metrics]
            )
                
        with gr.Tab("Model Evaluation"):
            with gr.Row():
                with gr.Column(scale=1):
                    eval_disaster_type = gr.Dropdown(
                        ["Earthquake", "Hurricane", "Flood"], 
                        label="Disaster Type",
                        value="Earthquake"
                    )
                    eval_model_type = gr.Dropdown(
                        ["Random Forest", "LSTM"], 
                        label="Model Type",
                        value="Random Forest"
                    )
                
                with gr.Column(scale=1):
                    test_data_option = gr.Radio(
                        ["Use sample test data", "Upload custom test data"], 
                        label="Test Data Source",
                        value="Use sample test data"
                    )
                    upload_data = gr.File(label="Upload Test Data (CSV)")
            
            eval_button = gr.Button("Evaluate Model")
            
            with gr.Row():
                with gr.Column(scale=2):
                    eval_plot = gr.Plot(label="Model Performance")
                with gr.Column(scale=1):
                    eval_metrics = gr.Dataframe(label="Performance Metrics")
                    eval_details = gr.JSON(label="Detailed Evaluation Results")
            
            eval_button.click(
                evaluate_model,
                [eval_disaster_type, eval_model_type, upload_data],
                [eval_plot, eval_metrics, eval_details]
            )
                
        with gr.Tab("Documentation"):
            gr.Markdown("""
            ## Model Testing Documentation
            
            ### Prediction Models
            
            - **Random Forest**: An ensemble learning method using multiple decision trees for prediction
            - **LSTM (Long Short-Term Memory)**: A recurrent neural network architecture for time series prediction
            
            ### Data Parameters
            
            Each disaster type uses different parameters for prediction:
            
            **Earthquake**
            - Location (latitude/longitude)
            - Historical seismic activity
            - Magnitude
            - Depth
            - Tectonic plate boundaries
            
            **Hurricane**
            - Sea surface temperature
            - Wind patterns
            - Air pressure
            - Historical storm tracks
            - Season/time of year
            
            **Flood**
            - Precipitation data
            - River levels
            - Terrain elevation
            - Soil saturation
            - Historical flood patterns
            
            ### Confidence Metrics
            
            Model confidence is calculated based on:
            - Data quality and quantity
            - Historical prediction accuracy
            - Parameter stability
            - Seasonal reliability
            
            ### Using This Interface
            
            1. Select the disaster type and model you want to test
            2. Configure the parameters or upload test data
            3. Run the prediction or evaluation
            4. Analyze the results and performance metrics
            """)
            
        with gr.Row():
            gr.Markdown("### Return to the main application")
            app_link = gr.Button("Go to Main Application")
            
            def open_main_app():
                # This is a workaround since we can't directly navigate
                return "Open the main app at: http://localhost:5000/"
                
            app_link.click(open_main_app, [], gr.Textbox())
    
    return interface

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=5000)
else:
    # For importing as a module from Streamlit
    interface = create_interface()
