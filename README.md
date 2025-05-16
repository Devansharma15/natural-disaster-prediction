📌 Project Title: Real-Time Natural Disaster Prediction System

Overview:
This project uses machine learning and geospatial data to forecast and visualize natural disasters such as earthquakes, floods, and hurricanes in real time.

*Key Features:

1.Interactive map and dashboard
2.Real-time disaster monitoring
3.Predictive analysis (7-day forecasts)
4.Confidence scoring
5.Supports LSTM and Random Forest

*Tech Stack:

1.Streamlit, Pandas, Plotly, Folium
2.ML models (Random Forest, LSTM)
3.APIs: USGS, NOAA, NASA

Screenshots:
Images from the interface such as:

1.Dashboard overview - 
![image](https://github.com/user-attachments/assets/97d2d5e7-bb0b-475b-9515-5d38c0eb6be5)

2.Map with predictions - 
![image](https://github.com/user-attachments/assets/c3f2950e-e32a-4bc9-a60f-acfd2fe7053e) (Earthquake)
![image](https://github.com/user-attachments/assets/0c954980-d723-41ca-a7c6-fdea021bfe9d) (Hurricane)
![image](https://github.com/user-attachments/assets/e983b248-1380-44d9-be32-e2618652615c) (Flood)

3.Historical trend chart - 
![image](https://github.com/user-attachments/assets/7ba2f68a-c021-4d11-8999-161c524e3017)

4.Forecast graph - 
![image](https://github.com/user-attachments/assets/e1ab6e64-5e89-4661-99ad-4c33ba8b8995) (Prediction for flood and the model is 80% accurate)


*How to Run Locally:
git clone https://github.com/yourusername/natural-disaster-prediction.git
cd natural-disaster-prediction
pip install -r requirements.txt
streamlit run app.py
