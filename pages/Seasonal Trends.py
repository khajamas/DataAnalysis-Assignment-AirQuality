import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
from datetime import datetime
import traceback
import matplotlib.pyplot as plt


# Custom CSS for better styling
# Update the custom CSS section at the top of the code
st.markdown("""
<style>
    /* Base font size increase */
    html, body, .stApp, .stMarkdown, .stText, .stNumberInput, .stSelectbox, .stSlider {
        font-size: 18px !important;
    }
    
    /* Headers */
    h1 {
        font-size: 36px !important;
    }
    h2 {
        font-size: 30px !important;
    }
    h3 {
        font-size: 26px !important;
    }
    h4 {
        font-size: 22px !important;
    }
    
    /* Tables and charts */
    .stTable {
        font-size: 18px !important;
    }
    
    /* Prediction boxes */
    .prediction-box h2 {
        font-size: 32px !important;
    }
    .prediction-box h3 {
        font-size: 28px !important;
    }
    
    /* AQI Health Impact Chart specific styles */
    .aqi-impact-table {
        font-size: 20px !important;
    }
    .aqi-impact-table th {
        font-size: 22px !important;
        font-weight: bold !important;
    }
    .aqi-impact-table td {
        font-size: 20px !important;
    }
    
    /* Other existing styles remain the same */
    .prediction-box {
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
    .good { background-color: #55A84F; color: white; }
    .satisfactory { background-color: #A3C853; color: white; }
    .moderate { background-color: #FFF833; color: black; }
    .poor { background-color: #F29C33; color: white; }
    .very-poor { background-color: #E93F33; color: white; }
    .severe { background-color: #AF2D24; color: white; }
    .plot-container {
        border: 1px solid #e1e4e8;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .suggestion-box {
        background-color: #f0f7fb;
        border-left: 5px solid #2196F3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 3px;
        font-size: 18px !important;
    }
    .impact-factor {
        background-color: #fff3e0;
        border-left: 5px solid #FF9800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 3px;
        font-size: 18px !important;
    }
</style>
""", unsafe_allow_html=True)

# Global error message placeholder
error_placeholder = st.empty()

def show_error(message):
    """Display error message in a consistent location"""
    error_placeholder.error(f" {str(message)}")



@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None})
def load_data():
    try:
        df = pd.read_csv(r'city_day.csv')
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.tz_localize(None)
        
        # Remove rows with invalid dates
        df = df[df['Date'].notna()]
        
        # For display purposes, create a string version
        df['Date_Display'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Validate required columns
        required_columns = ['Date', 'City', 'AQI']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        # Convert and validate dates
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Date_Display'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Check for AQI values
        if df['AQI'].isna().all():
            raise ValueError("No valid AQI values found in dataset")
        df = df.dropna(subset=['AQI'])

        # Handle pollutants
        pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 
                         'Benzene', 'Toluene', 'Xylene']
        existing_pollutants = [col for col in pollutant_cols if col in df.columns]
        df = df.dropna(subset=existing_pollutants, how='all')

        # Fill missing values
        for col in existing_pollutants:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())

        return df
    
    except FileNotFoundError:
        raise FileNotFoundError("Data file not found. Please check the file path.")
    except pd.errors.EmptyDataError:
        raise ValueError("The data file is empty or corrupted.")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


try:
        st.title("Seasonal Trends")
        
        # Load data with error handling
        try:
            df = load_data()
        except Exception as e:
            show_error(f"Data loading failed: {str(e)}")
            st.stop()
   
except Exception as e:
    show_error(f"Application Error: {str(e)}")
    st.error(traceback.format_exc())
    

# Handle pollutants
pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 
                         'Benzene', 'Toluene', 'Xylene']
existing_pollutants = [col for col in pollutant_cols if col in df.columns]
df = df.dropna(subset=existing_pollutants, how='all')

# Fill missing values
for col in existing_pollutants:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mean())








# Historical data analysis
st.header("Historical Trends & Seasonal Analysis")
try:
    col1, col2 = st.columns(2)
            
    with col1:
        trend_city = st.selectbox("Select City for Trends", df['City'].unique())
            
    with col2:
        trend_year = st.selectbox("Select Year", sorted(df['Year'].unique(), reverse=True))
            
    city_year_df = df.loc[(df['City'] == trend_city) & (df['Year'] == trend_year)]
            
    if not city_year_df.empty:
        # Monthly trends for the selected year and city
        monthly_avg = city_year_df.groupby('Month', observed=True)['AQI'].mean().reset_index()
        monthly_avg['Month_Name'] = monthly_avg['Month'].apply(lambda x: datetime(2000, x, 1).strftime('%b'))
                
        fig = px.line(
                    monthly_avg, 
                    x='Month_Name', 
                    y='AQI',
                    markers=True,
                    title=f"Monthly AQI Trends for {trend_city} in {trend_year}"
                )
        st.plotly_chart(fig, use_container_width=True)
                
                # Seasonal patterns
        def get_season(month):
                    if month in [12, 1, 2]:
                        return 'Winter'
                    elif month in [3, 4, 5]:
                        return 'Spring'
                    elif month in [6, 7, 8]:
                        return 'Summer'
                    else:
                        return 'Fall'
                
        city_year_df['Season'] = city_year_df['Month'].apply(get_season)
        seasonal_avg = city_year_df.groupby('Season')['AQI'].mean().reset_index()
                
                # Order seasons properly - only include seasons that exist in the data
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        existing_seasons = [s for s in season_order if s in seasonal_avg['Season'].values]
                
        seasonal_avg['Season'] = pd.Categorical(
                    seasonal_avg['Season'], 
                    categories=existing_seasons,  # Only use existing seasons
                    ordered=True
                )
        seasonal_avg = seasonal_avg.sort_values('Season')
                
                # fig = px.bar(
                #     seasonal_avg,
                #     x='Season',
                #     y='AQI',
                #     color='Season',
                #     title=f"Seasonal AQI Patterns for {trend_city} in {trend_year}"
                # )
                # st.plotly_chart(fig, use_container_width=True)
        fig = px.bar(
                    seasonal_avg,
                    x='Season',
                    y='AQI',
                    color='Season',
                    text='AQI',
                    title=f"Seasonal AQI Patterns for {trend_city} in {trend_year}",
                    category_orders={"Season": ['Winter', 'Spring', 'Summer', 'Fall']}
                )

                # Customize bar appearance and labels
        fig.update_traces(
                    width=0.7,  # Adjusted bar width
                    texttemplate='%{y:.1f}',
                    textposition='inside',
                    insidetextanchor='middle',  # Ensures perfect vertical centering
                    marker_line_color='rgb(0,0,0)',
                    marker_line_width=1,
                    opacity=0.9
                )

                # Critical fix for label alignment
        fig.update_layout(
                    xaxis = dict(
                        title="Season",
                        tickmode = 'array',  # Explicit tick positioning
                        tickvals = [0, 1, 2, 3],  # Matches the 4 seasons
                        ticktext = ['Winter', 'Spring', 'Summer', 'Fall'],  # Explicit labels
                        tickangle = 0,
                        tickfont = dict(size=14)
                    ),
                    yaxis_title="Average AQI",
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=50, r=50, t=80, b=80)
                )

                # Add vertical grid lines to visually confirm alignment
        fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(211, 211, 211, 0.3)'
                )

        st.plotly_chart(fig, use_container_width=True)
                # Pollutant composition by season
        if existing_pollutants:
            seasonal_pollutants = city_year_df.groupby('Season')[existing_pollutants].mean().reset_index()
                    
                    # Only include seasons that exist in the data
            seasonal_pollutants['Season'] = pd.Categorical(
                        seasonal_pollutants['Season'], 
                        categories=existing_seasons,
                        ordered=True
                    )
            seasonal_pollutants = seasonal_pollutants.sort_values('Season')
                    
                    # Normalize data for better visualization
            for col in existing_pollutants:
                        max_val = seasonal_pollutants[col].max()
                        if max_val > 0:
                            seasonal_pollutants[f'{col}_norm'] = seasonal_pollutants[col] / max_val
                    
            norm_cols = [f'{col}_norm' for col in existing_pollutants]
                    
                    # Create radar chart only if we have data
            if not seasonal_pollutants.empty:
                fig = go.Figure()
                        
                for season in existing_seasons:
                    season_data = seasonal_pollutants[seasonal_pollutants['Season'] == season]
                    if not season_data.empty:
                        fig.add_trace(go.Scatterpolar(
                                    r=season_data[norm_cols].values.flatten().tolist(),
                                    theta=existing_pollutants,
                                    fill='toself',
                                    name=season
                                ))
                        
                    fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title=f"Seasonal Pollutant Composition for {trend_city} in {trend_year}"
                        )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for {trend_city} in {trend_year}")

except Exception as e:
        show_error(f"Error in historical trend analysis: {str(e)}")
