import streamlit as st
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Air Quality Analysis & Prediction",
    page_icon="🌍",
    layout="wide"
)


st.markdown(
    """
    <style>
    .title-text {
        font-size: 48px;
        font-weight: 700;
        color: #2c3e50;
    }
    .subtitle-text {
        font-size: 20px;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hero Section
st.markdown('<div class="title-text">🌍 Air Quality Analysis & AQI Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Data-driven insights into air pollution trends across Indian cities (2015–2020)</div>',
    unsafe_allow_html=True
)

st.divider()

# About Section
st.header("📌 About the Assignment")
st.write("""
Air pollution is a major environmental and public health concern in India.
This application analyzes air quality data collected from multiple Indian cities
between **2015 and 2020**, focusing on key pollutants and their impact on the **Air Quality Index (AQI)**.
""")

# Dataset
st.header("📊 Dataset Overview")
st.write("""
- Time Period: **2015 – 2020**  
- Coverage: **Multiple Indian cities**  
- Pollutants: PM2.5,PM10,NO,NO2,NOx,NH3,CO,SO2,O3,Benzene,Toluene,Xylene 
- Target Variable: **AQI**
""")

# Features
st.header("⚙️ Application Features")
st.markdown("""
✔ City-wise and year-wise AQI analysis  
✔ Pollutant distribution and correlation analysis  
✔ Interactive visualizations  
✔ AQI prediction using **Random Forest Machine Learning Model**
""")

# Importance
st.header("🚨 Why Air Quality Monitoring Matters")
st.write("""
Poor air quality leads to severe health risks, environmental damage,
and reduced quality of life. This project aims to provide **actionable insights**
through data analysis and predictive modeling.
""")

# Tech Stack
st.header("🛠️ Technologies Used")
st.markdown("""
- Python  
- Pandas, NumPy  
- Plotly, Matplotlib, Seaborn  
- Scikit-learn  
- Streamlit  
""")

st.divider()





st.success("⬅️ Use the sidebar to explore analysis dashboards and AQI prediction modules.")
