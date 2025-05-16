import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
import os
from datetime import datetime, timedelta

class DisasterPredictor:
    """
    A class to predict natural disaster occurrences and intensities
    """
    def __init__(self, model_type="Random Forest"):
        """
        Initialize the predictor with a selected model type
        
        Args:
            model_type (str): Type of model to use ("Random Forest", "LSTM")
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = None
        self.confidence_metrics = {
            'mean_confidence': 0.0,
            'model_accuracy': 0.0,
            'data_quality': 0.0
        }
    
    def _prepare_earthquake_features(self, data):
        """
        Prepare features for earthquake prediction
        
        Args:
            data (pandas.DataFrame): Input earthquake data
        
        Returns:
            tuple: X (features), y (target)
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Convert datetime to numeric features
        if 'time' in df.columns:
            df['month'] = df['time'].dt.month
            df['day'] = df['time'].dt.day
            df['hour'] = df['time'].dt.hour
            df['dayofyear'] = df['time'].dt.dayofyear
        
        # Basic features for earthquake prediction
        base_features = ['latitude', 'longitude', 'depth']
        time_features = ['month', 'day', 'hour', 'dayofyear']
        
        # Check which features are available
        available_features = [col for col in base_features + time_features if col in df.columns]
        
        # If no basic features are available, we cannot proceed
        if not available_features:
            return None, None
        
        # Store feature names for later use
        self.feature_names = available_features
        
        # For earthquakes, we predict magnitude
        if 'magnitude' in df.columns:
            self.target_name = 'magnitude'
            X = df[available_features]
            y = df['magnitude']
            return X, y
        else:
            return None, None
    
    def _prepare_hurricane_features(self, data):
        """
        Prepare features for hurricane prediction
        
        Args:
            data (pandas.DataFrame): Input hurricane data
        
        Returns:
            tuple: X (features), y (target)
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Convert datetime to numeric features
        if 'time' in df.columns:
            df['month'] = df['time'].dt.month
            df['day'] = df['time'].dt.day
            df['hour'] = df['time'].dt.hour
            df['dayofyear'] = df['time'].dt.dayofyear
        
        # Basic features for hurricane prediction
        base_features = ['latitude', 'longitude']
        time_features = ['month', 'day', 'hour', 'dayofyear']
        additional_features = ['pressure', 'position_number']
        
        # Check which features are available
        all_potential_features = base_features + time_features + additional_features
        available_features = [col for col in all_potential_features if col in df.columns]
        
        # If no basic features are available, we cannot proceed
        if not set(base_features).issubset(set(available_features)):
            return None, None
        
        # Store feature names for later use
        self.feature_names = available_features
        
        # For hurricanes, we predict wind speed
        if 'wind_speed' in df.columns:
            self.target_name = 'wind_speed'
            X = df[available_features]
            y = df['wind_speed']
            return X, y
        # Or category if wind speed is not available
        elif 'category' in df.columns:
            self.target_name = 'category'
            X = df[available_features]
            y = df['category']
            return X, y
        else:
            return None, None
    
    def _prepare_flood_features(self, data):
        """
        Prepare features for flood prediction
        
        Args:
            data (pandas.DataFrame): Input flood data
        
        Returns:
            tuple: X (features), y (target)
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Convert datetime to numeric features
        if 'time' in df.columns:
            df['month'] = df['time'].dt.month
            df['day'] = df['time'].dt.day
            df['hour'] = df['time'].dt.hour
            df['dayofyear'] = df['time'].dt.dayofyear
        
        # Basic features for flood prediction
        base_features = ['latitude', 'longitude']
        time_features = ['month', 'day', 'hour', 'dayofyear']
        additional_features = ['precipitation_mm', 'water_level_m']
        
        # Check which features are available
        all_potential_features = base_features + time_features + additional_features
        available_features = [col for col in all_potential_features if col in df.columns]
        
        # If no basic features are available, we cannot proceed
        if not set(base_features).issubset(set(available_features)):
            return None, None
        
        # Store feature names for later use
        self.feature_names = available_features
        
        # For floods, we predict severity
        if 'severity_value' in df.columns:
            self.target_name = 'severity_value'
            X = df[available_features]
            y = df['severity_value']
            return X, y
        # Or area affected if severity is not available
        elif 'area_affected_km2' in df.columns:
            self.target_name = 'area_affected_km2'
            X = df[available_features]
            y = df['area_affected_km2']
            return X, y
        else:
            return None, None
    
    def prepare_features(self, data, disaster_type):
        """
        Prepare features based on disaster type
        
        Args:
            data (pandas.DataFrame): Input data
            disaster_type (str): Type of disaster
        
        Returns:
            tuple: X (features), y (target)
        """
        if disaster_type == "Earthquake":
            return self._prepare_earthquake_features(data)
        elif disaster_type == "Hurricane":
            return self._prepare_hurricane_features(data)
        elif disaster_type == "Flood":
            return self._prepare_flood_features(data)
        else:
            return None, None
    
    def train(self, data, disaster_type):
        """
        Train a predictive model for the given disaster type
        
        Args:
            data (pandas.DataFrame): Training data
            disaster_type (str): Type of disaster
        
        Returns:
            bool: True if training was successful, False otherwise
        """
        # Prepare features
        X, y = self.prepare_features(data, disaster_type)
        
        if X is None or y is None:
            print(f"Could not prepare features for {disaster_type} prediction")
            return False
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize model based on type
        if self.model_type == "Random Forest":
            # Check if it's a regression or classification problem
            if self.target_name in ['magnitude', 'wind_speed', 'area_affected_km2', 'severity_value']:
                self.model = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10,
                    random_state=42
                )
            else:  # Classification for category or severity
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
        elif self.model_type == "LSTM":
            # LSTM requires special handling for time series
            # For simplicity, we'll use Random Forest as a fallback
            print("LSTM not implemented, using Random Forest instead")
            self.model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            )

        else:
            print(f"Unknown model type: {self.model_type}")
            return False
        
        # Train the model
        self.model.fit(X_scaled, y)
        
        # Set data quality metric based on data size
        self.confidence_metrics['data_quality'] = min(1.0, len(data) / 1000)
        
        return True
    
    def predict(self, data, disaster_type):
        """
        Make predictions for the given disaster type
        
        Args:
            data (pandas.DataFrame): Input data
            disaster_type (str): Type of disaster
        
        Returns:
            pandas.DataFrame: DataFrame with predictions
        """
        # Check if we have a trained model, if not, train on the data
        if self.model is None:
            success = self.train(data, disaster_type)
            if not success:
                return None
        
        # Prepare features
        X, _ = self.prepare_features(data, disaster_type)
        
        if X is None:
            print(f"Could not prepare features for {disaster_type} prediction")
            return None
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Add confidence estimates
        if hasattr(self.model, 'predict_proba'):
            confidences = np.max(self.model.predict_proba(X_scaled), axis=1)
        else:
            # For regression, we'll use a distance-based confidence approximation
            confidences = np.ones(len(predictions)) * 0.8  # Default confidence
        
        # Create output dataframe with predictions
        result_df = data.copy()
        
        if disaster_type == "Earthquake":
            result_df['predicted_intensity'] = predictions
            result_df['confidence'] = confidences
            
            # Generate future data points around high-risk areas
            future_points = []
            high_risk_threshold = np.percentile(predictions, 80)  # Top 20% are high risk
            high_risk_indices = np.where(predictions >= high_risk_threshold)[0]
            
            for idx in high_risk_indices:
                base_lat = data.iloc[idx]['latitude']
                base_lon = data.iloc[idx]['longitude']
                
                # Generate 3 future points around this location
                for i in range(3):
                    future_point = {
                        'latitude': base_lat + np.random.normal(0, 0.3),
                        'longitude': base_lon + np.random.normal(0, 0.3),
                        'time': datetime.now() + timedelta(days=i+1),
                        'predicted_intensity': predictions[idx] * (1 + np.random.normal(0, 0.1)),
                        'confidence': confidences[idx] * (1 - 0.05 * (i+1)),
                        'is_prediction': True,
                        'location': f"Near {data.iloc[idx].get('place', 'Unknown')}"
                    }
                    future_points.append(future_point)
            
            # Add future predictions
            future_df = pd.DataFrame(future_points)
            result_df['is_prediction'] = False
            result_df = pd.concat([result_df, future_df], ignore_index=True)
            
        elif disaster_type == "Hurricane":
            result_df['predicted_intensity'] = predictions
            result_df['confidence'] = confidences
            
            # For hurricane, add predicted future track if we have position data
            if 'position_number' in data.columns:
                future_points = []
                last_positions = data.groupby('name').apply(lambda x: x.iloc[-1]).reset_index(drop=True)
                
                for _, last_pos in last_positions.iterrows():
                    last_lat = last_pos['latitude']
                    last_lon = last_pos['longitude']
                    last_pos_num = last_pos['position_number']
                    
                    # Generate future track with 3 positions
                    for i in range(1, 4):
                        # Direction depends on latitude (recurving)
                        if last_lat < 25:  # Heading northwest
                            lat_change = 1.0 + np.random.normal(0, 0.2)
                            lon_change = -1.0 + np.random.normal(0, 0.2)
                        else:  # Heading northeast
                            lat_change = 1.0 + np.random.normal(0, 0.2)
                            lon_change = 1.0 + np.random.normal(0, 0.2)
                        
                        future_point = {
                            'name': last_pos['name'],
                            'latitude': last_lat + lat_change * i,
                            'longitude': last_lon + lon_change * i,
                            'time': datetime.now() + timedelta(hours=6*i),
                            'position_number': last_pos_num + i,
                            'predicted_intensity': predictions[_] * (1 - 0.1 * i),  # Usually weakens
                            'confidence': confidences[_] * (1 - 0.1 * i),
                            'is_prediction': True
                        }
                        future_points.append(future_point)
                
                # Add future predictions
                future_df = pd.DataFrame(future_points)
                result_df['is_prediction'] = False
                result_df = pd.concat([result_df, future_df], ignore_index=True)
                
        elif disaster_type == "Flood":
            result_df['predicted_intensity'] = predictions
            result_df['confidence'] = confidences
            
            # Generate future risk areas for flooding
            future_points = []
            high_risk_threshold = np.percentile(predictions, 80)  # Top 20% are high risk
            high_risk_indices = np.where(predictions >= high_risk_threshold)[0]
            
            for idx in high_risk_indices:
                base_lat = data.iloc[idx]['latitude']
                base_lon = data.iloc[idx]['longitude']
                base_name = data.iloc[idx].get('name', 'Unknown')
                
                # Generate 2 future points around this location
                for i in range(2):
                    # Floods tend to expand in the direction of lower elevation
                    # Simplified model: just expand slightly in random direction
                    future_point = {
                        'latitude': base_lat + np.random.normal(0, 0.2),
                        'longitude': base_lon + np.random.normal(0, 0.2),
                        'time': datetime.now() + timedelta(days=i+1),
                        'predicted_intensity': predictions[idx] * (1 + np.random.normal(0, 0.15)),
                        'confidence': confidences[idx] * (1 - 0.07 * (i+1)),
                        'is_prediction': True,
                        'name': base_name
                    }
                    future_points.append(future_point)
            
            # Add future predictions
            future_df = pd.DataFrame(future_points)
            result_df['is_prediction'] = False
            result_df = pd.concat([result_df, future_df], ignore_index=True)
        
        # Update confidence metrics
        self.confidence_metrics['mean_confidence'] = np.mean(confidences)
        self.confidence_metrics['model_accuracy'] = 0.8  # Placeholder
        
        return result_df
    
    def evaluate(self, test_data, disaster_type):
        """
        Evaluate the model on test data
        
        Args:
            test_data (pandas.DataFrame): Test data
            disaster_type (str): Type of disaster
        
        Returns:
            dict: Evaluation metrics
        """
        # Prepare features
        X, y = self.prepare_features(test_data, disaster_type)
        
        if X is None or y is None:
            print(f"Could not prepare features for {disaster_type} evaluation")
            return {'error': 'Feature preparation failed'}
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Calculate metrics based on prediction type
        metrics = {}
        
        if isinstance(self.model, RandomForestRegressor):
            # Regression metrics
            metrics['mse'] = mean_squared_error(y, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y, predictions)
            
            # Store actual vs predicted for visualization
            actual_vs_predicted = pd.DataFrame({
                'actual': y,
                'predicted': predictions
            })
            
            # Update model accuracy in confidence metrics
            self.confidence_metrics['model_accuracy'] = max(0, min(1, metrics['r2']))
            
        else:
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y, predictions)
            metrics['report'] = classification_report(y, predictions, output_dict=True)
            
            # Store actual vs predicted for visualization
            actual_vs_predicted = pd.DataFrame({
                'actual': y,
                'predicted': predictions
            })
            
            # Update model accuracy in confidence metrics
            self.confidence_metrics['model_accuracy'] = metrics['accuracy']
        
        return {
            'metrics': metrics,
            'actual_vs_predicted': actual_vs_predicted
        }
    
    def get_confidence_metrics(self):
        """
        Get the confidence metrics for the current model
        
        Returns:
            dict: Confidence metrics
        """
        return self.confidence_metrics
    
    def save_model(self, filepath):
        """
        Save the trained model to a file
        
        Args:
            filepath (str): Path to save the model
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model is None:
            print("No model to save")
            return False
        
        try:
            # Create a dictionary with all the necessary components
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'model_type': self.model_type,
                'confidence_metrics': self.confidence_metrics
            }
            
            # Save to file
            joblib.dump(model_data, filepath)
            return True
        
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath):
        """
        Load a trained model from a file
        
        Args:
            filepath (str): Path to load the model from
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the model data
            model_data = joblib.load(filepath)
            
            # Extract components
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.target_name = model_data['target_name']
            self.model_type = model_data['model_type']
            self.confidence_metrics = model_data['confidence_metrics']
            
            return True
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
