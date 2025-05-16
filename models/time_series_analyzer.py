import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

class TimeSeriesAnalyzer:
    """
    A class for analyzing time series data related to natural disasters
    """
    
    def __init__(self):
        """Initialize the TimeSeriesAnalyzer"""
        self.data = None
        self.disaster_type = None
        self.decomposition = None
        self.trend_metrics = None
    
    def analyze_trends(self, data, disaster_type):
        """
        Analyze trends in historical disaster data
        
        Args:
            data (pandas.DataFrame): Historical disaster data with 'date' column
            disaster_type (str): Type of disaster (Earthquake, Hurricane, Flood)
        
        Returns:
            dict: Trend metrics and analysis results
        """
        self.data = data
        self.disaster_type = disaster_type
        
        # Verify that we have date column and enough data
        if 'date' not in data.columns:
            return None
        
        if len(data) < 10:
            return {"Status": "Insufficient data for trend analysis"}
        
        # Determine the value column based on disaster type
        value_column = self._get_value_column(disaster_type)
        
        if value_column not in data.columns:
            return None
        
        # Aggregate data by month for smoother trends
        monthly_data = self._aggregate_monthly(data, value_column)
        
        # Calculate basic trend metrics
        self.trend_metrics = self._calculate_trend_metrics(monthly_data, value_column)
        
        # Try to perform time series decomposition if we have enough data
        try:
            if len(monthly_data) >= 12:  # Need at least a year of data for seasonal decomposition
                self.decomposition = self._decompose_time_series(monthly_data, value_column)
                
                # Add decomposition results to metrics
                if self.decomposition is not None:
                    self.trend_metrics["Trend Strength"] = self._calculate_trend_strength()
                    self.trend_metrics["Seasonality Strength"] = self._calculate_seasonality_strength()
        except Exception as e:
            print(f"Error during time series decomposition: {e}")
            # Continue without decomposition results
        
        # Try to perform stationarity test
        try:
            adf_result = adfuller(monthly_data[value_column].fillna(method='ffill'))
            self.trend_metrics["Is Stationary"] = "Yes" if adf_result[1] < 0.05 else "No"
            self.trend_metrics["Stationarity p-value"] = round(adf_result[1], 3)
        except Exception as e:
            print(f"Error during stationarity test: {e}")
            # Continue without stationarity results
        
        return self.trend_metrics
    
    def _get_value_column(self, disaster_type):
        """
        Get the appropriate value column for the given disaster type
        
        Args:
            disaster_type (str): Type of disaster
        
        Returns:
            str: Column name for the value to analyze
        """
        if disaster_type == "Earthquake":
            return "magnitude"
        elif disaster_type == "Hurricane":
            if "wind_speed" in self.data.columns:
                return "wind_speed"
            else:
                return "active_count"
        elif disaster_type == "Flood":
            if "severity_value" in self.data.columns:
                return "severity_value"
            elif "area_affected" in self.data.columns:
                return "area_affected"
            else:
                return "casualties"
        else:
            return "value"  # Default fallback
    
    def _aggregate_monthly(self, data, value_column):
        """
        Aggregate data by month
        
        Args:
            data (pandas.DataFrame): Input data with 'date' column
            value_column (str): Column to aggregate
        
        Returns:
            pandas.DataFrame: Monthly aggregated data
        """
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
        
        # Set date as index
        df = data.set_index('date')
        
        # Resample to monthly frequency using appropriate aggregation
        if value_column in ['magnitude', 'wind_speed', 'severity_value']:
            # For intensity measures, use mean
            monthly = df[value_column].resample('MS').mean().reset_index()
        elif value_column in ['active_count', 'casualties']:
            # For count measures, use sum
            monthly = df[value_column].resample('MS').sum().reset_index()
        else:
            # For other measures, use mean as default
            monthly = df[value_column].resample('MS').mean().reset_index()
        
        return monthly
    
    def _calculate_trend_metrics(self, monthly_data, value_column):
        """
        Calculate basic trend metrics
        
        Args:
            monthly_data (pandas.DataFrame): Monthly aggregated data
            value_column (str): Column with values
        
        Returns:
            dict: Trend metrics
        """
        metrics = {}
        
        # Calculate overall change metrics
        start_value = monthly_data[value_column].iloc[0]
        end_value = monthly_data[value_column].iloc[-1]
        
        # Avoid division by zero
        if start_value == 0:
            start_value = 0.01
        
        metrics["Overall Change"] = f"{(end_value - start_value):.2f}"
        metrics["Percent Change"] = f"{((end_value - start_value) / start_value * 100):.1f}%"
        
        # Calculate recent trend (last 6 months or all if less than 6)
        recent_months = min(6, len(monthly_data))
        recent_data = monthly_data.iloc[-recent_months:]
        
        if len(recent_data) >= 2:
            recent_start = recent_data[value_column].iloc[0]
            recent_end = recent_data[value_column].iloc[-1]
            
            # Avoid division by zero
            if recent_start == 0:
                recent_start = 0.01
            
            metrics["Recent Trend"] = "Increasing" if recent_end > recent_start else "Decreasing"
            metrics["Recent Change Rate"] = f"{((recent_end - recent_start) / recent_start * 100):.1f}% per period"
        
        # Calculate year-over-year change if we have enough data
        if len(monthly_data) >= 13:
            current = monthly_data[value_column].iloc[-1]
            year_ago = monthly_data[value_column].iloc[-13]
            
            # Avoid division by zero
            if year_ago == 0:
                year_ago = 0.01
            
            metrics["Year-over-Year Change"] = f"{((current - year_ago) / year_ago * 100):.1f}%"
        
        # Calculate average value
        metrics["Average Value"] = f"{monthly_data[value_column].mean():.2f}"
        
        # Calculate volatility (standard deviation)
        metrics["Volatility"] = f"{monthly_data[value_column].std():.2f}"
        
        return metrics
    
    def _decompose_time_series(self, monthly_data, value_column):
        """
        Decompose time series into trend, seasonal, and residual components
        
        Args:
            monthly_data (pandas.DataFrame): Monthly aggregated data
            value_column (str): Column with values
        
        Returns:
            statsmodels.tsa.seasonal.DecomposeResult: Decomposition result
        """
        # Fill missing values if any
        data = monthly_data.copy()
        data[value_column] = data[value_column].fillna(method='ffill').fillna(method='bfill')
        
        # Need a DatetimeIndex for decomposition
        data = data.set_index('date')
        
        # Apply seasonal decomposition
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                # Try additive decomposition first
                decomposition = seasonal_decompose(data[value_column], model='additive', period=12)
                return decomposition
            except:
                try:
                    # If additive fails, try multiplicative
                    # Ensure no zeros or negative values for multiplicative model
                    if (data[value_column] <= 0).any():
                        data[value_column] = data[value_column] - data[value_column].min() + 0.1
                    
                    decomposition = seasonal_decompose(data[value_column], model='multiplicative', period=12)
                    return decomposition
                except Exception as e:
                    print(f"Decomposition failed: {e}")
                    return None
    
    def _calculate_trend_strength(self):
        """
        Calculate the strength of the trend component
        
        Returns:
            float: Trend strength (0-1 scale)
        """
        if self.decomposition is None:
            return None
        
        # Trend strength = 1 - (Variance of residual / Variance of deseasonalized series)
        var_resid = np.var(self.decomposition.resid.dropna())
        var_deseason = np.var((self.decomposition.trend + self.decomposition.resid).dropna())
        
        if var_deseason == 0:
            return 0
        
        trend_strength = max(0, min(1, 1 - (var_resid / var_deseason)))
        return round(trend_strength, 2)
    
    def _calculate_seasonality_strength(self):
        """
        Calculate the strength of the seasonal component
        
        Returns:
            float: Seasonality strength (0-1 scale)
        """
        if self.decomposition is None:
            return None
        
        # Seasonal strength = 1 - (Variance of residual / Variance of detrended series)
        var_resid = np.var(self.decomposition.resid.dropna())
        var_detrend = np.var((self.decomposition.seasonal + self.decomposition.resid).dropna())
        
        if var_detrend == 0:
            return 0
        
        seasonal_strength = max(0, min(1, 1 - (var_resid / var_detrend)))
        return round(seasonal_strength, 2)
    
    def plot_seasonality(self, data, disaster_type):
        """
        Plot seasonality patterns in the data
        
        Args:
            data (pandas.DataFrame): Historical disaster data
            disaster_type (str): Type of disaster
        
        Returns:
            plotly.graph_objects.Figure: Seasonality plot
        """
        # Ensure we have the data loaded
        if self.data is None or not np.array_equal(self.data, data):
            self.data = data
            self.disaster_type = disaster_type
        
        value_column = self._get_value_column(disaster_type)
        
        if value_column not in data.columns:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No suitable data available for seasonality analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
        
        # Extract month and year for grouping
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        
        # Group by month and calculate average value
        monthly_avg = data.groupby('month')[value_column].mean().reset_index()
        
        # Get month names for the x-axis
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_avg['month_name'] = monthly_avg['month'].apply(lambda x: month_names[x-1])
        
        # Create the plot
        title_map = {
            "magnitude": "Average Earthquake Magnitude by Month",
            "wind_speed": "Average Hurricane Wind Speed by Month",
            "active_count": "Average Hurricane Count by Month",
            "severity_value": "Average Flood Severity by Month",
            "area_affected": "Average Flood Area Affected by Month",
            "casualties": "Average Casualties by Month"
        }
        
        title = title_map.get(value_column, f"Average {value_column} by Month")
        
        fig = px.line(
            monthly_avg,
            x='month',
            y=value_column,
            markers=True,
            title=title,
            labels={
                value_column: value_column.replace('_', ' ').title(),
                'month': 'Month'
            }
        )
        
        # Add month names to x-axis
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=month_names
            )
        )
        
        # Add range shading for typical high season if appropriate
        if disaster_type == "Hurricane":
            # Hurricane season is typically June through November
            fig.add_vrect(
                x0=6, x1=11.5,
                fillcolor="rgba(255, 0, 0, 0.1)", opacity=0.5,
                layer="below", line_width=0,
                annotation_text="Hurricane Season",
                annotation_position="top left"
            )
        elif disaster_type == "Flood":
            # Flood season depends on region, but often spring/summer
            peak_month = monthly_avg[value_column].idxmax()
            peak = monthly_avg.loc[peak_month, 'month']
            
            # Highlight 3 months around the peak
            start = max(1, peak - 1)
            end = min(12, peak + 1)
            
            fig.add_vrect(
                x0=start-0.5, x1=end+0.5,
                fillcolor="rgba(0, 0, 255, 0.1)", opacity=0.5,
                layer="below", line_width=0,
                annotation_text="Peak Season",
                annotation_position="top left"
            )
        
        return fig
    
    def forecast_arima(self, data, disaster_type, forecast_periods=6):
        """
        Generate forecast using ARIMA model
        
        Args:
            data (pandas.DataFrame): Historical disaster data
            disaster_type (str): Type of disaster
            forecast_periods (int): Number of periods to forecast
        
        Returns:
            tuple: (DataFrame with forecasts, plotly Figure)
        """
        # Ensure we have the data loaded
        if self.data is None or not np.array_equal(self.data, data):
            self.data = data
            self.disaster_type = disaster_type
        
        value_column = self._get_value_column(disaster_type)
        
        if value_column not in data.columns or len(data) < 10:
            # Not enough data for forecasting
            return None, None
        
        # Aggregate to monthly
        monthly_data = self._aggregate_monthly(data, value_column)
        
        # Fit ARIMA model
        try:
            # Simple order selection (in practice, would use auto_arima or more sophisticated approach)
            if len(monthly_data) >= 24:  # If we have at least 2 years of data
                order = (1, 1, 1)  # Simple differencing model
                seasonal_order = (1, 1, 1, 12)  # Simple seasonal model with yearly seasonality
                model = ARIMA(
                    monthly_data[value_column], 
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                # With less data, use simpler model
                order = (1, 1, 0)
                model = ARIMA(monthly_data[value_column], order=order)
            
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(steps=forecast_periods)
            forecast_index = pd.date_range(
                start=monthly_data['date'].iloc[-1] + pd.DateOffset(months=1),
                periods=forecast_periods,
                freq='MS'
            )
            
            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                'date': forecast_index,
                value_column: forecast
            })
            
            # Create confidence intervals (simplified)
            confidence = 1.96 * model_fit.params[-1]  # Using residual standard error for simplicity
            forecast_df[f'{value_column}_lower'] = forecast_df[value_column] - confidence
            forecast_df[f'{value_column}_upper'] = forecast_df[value_column] + confidence
            
            # Create plot with historical and forecast data
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=monthly_data['date'],
                y=monthly_data[value_column],
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df[value_column],
                name='Forecast',
                line=dict(color='red', dash='dot')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                y=forecast_df[f'{value_column}_upper'].tolist() + forecast_df[f'{value_column}_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0)'),
                hoverinfo="skip",
                showlegend=False
            ))
            
            title_map = {
                "magnitude": f"Earthquake Magnitude Forecast (Next {forecast_periods} Months)",
                "wind_speed": f"Hurricane Wind Speed Forecast (Next {forecast_periods} Months)",
                "active_count": f"Hurricane Activity Forecast (Next {forecast_periods} Months)",
                "severity_value": f"Flood Severity Forecast (Next {forecast_periods} Months)",
                "area_affected": f"Flood Area Forecast (Next {forecast_periods} Months)",
                "casualties": f"Casualties Forecast (Next {forecast_periods} Months)"
            }
            
            fig.update_layout(
                title=title_map.get(value_column, f"{value_column.title()} Forecast"),
                xaxis_title="Date",
                yaxis_title=value_column.replace('_', ' ').title(),
                legend_title="Data Type",
                hovermode="x unified"
            )
            
            return forecast_df, fig
            
        except Exception as e:
            print(f"Error in ARIMA forecasting: {e}")
            return None, None
