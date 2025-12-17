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
        st.title("Exploratory Data Analysis")
        
        # Load data with error handling
        try:
            df = load_data()
        except Exception as e:
            show_error(f"Data loading failed: {str(e)}")
            st.stop()
   
except Exception as e:
    show_error(f"Application Error: {str(e)}")
    st.error(traceback.format_exc())
    
    
#custom colors 
try:
    # Create AQI buckets if they don't exist
    df['AQI_Bucket'] = pd.cut(
        df['AQI'],
        bins=[0, 50, 100, 200, 300, 400, float('inf')],
        labels=['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    )
            
    aqi_color_map = {
                'Good': "#48D3BC", 
                'Satisfactory': "#49D82C", 
                'Moderate': '#FFF833',
                'Poor': '#F29C33', 
                'Very Poor': "#3096DA", 
                'Severe': '#AF2D24'
    }
            
    # Get counts and ensure all categories are present
    aqi_bucket_counts = df['AQI_Bucket'].value_counts().reset_index()
    aqi_bucket_counts.columns = ['AQI Category', 'Count']
            
    # Define the correct order
    category_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
            
    # Ensure all categories are present (fill with 0 if missing)
    for cat in category_order:
        if cat not in aqi_bucket_counts['AQI Category'].values:
            aqi_bucket_counts = pd.concat([
                        aqi_bucket_counts,
                        pd.DataFrame({'AQI Category': [cat], 'Count': [0]})
            ], ignore_index=True)
            
    # Convert to categorical with correct ordering
        aqi_bucket_counts['AQI Category'] = pd.Categorical(
                aqi_bucket_counts['AQI Category'],
                categories=category_order,
                ordered=True
        )
            
    # Sort by the categorical order
    aqi_bucket_counts = aqi_bucket_counts.sort_values('AQI Category')
    
    fig = px.bar(
                aqi_bucket_counts, 
                x='AQI Category', 
                y='Count', 
                color='AQI Category',
                color_discrete_map=aqi_color_map,
                title="Distribution of AQI Categories",
                category_orders={"AQI Category": category_order},
                width=800,  # Adjust overall width
                height=500  # Adjust height if needed
            )
except Exception as e:
            show_error(f"Error generating AQI category distribution: {str(e)}")   
            

#EDA

st.sidebar.header("🔍 Filter")

city_list = sorted(df['City'].dropna().unique())
default_city = "Ahmedabad" if "Ahmedabad" in city_list else city_list[0]

selected_city = st.sidebar.selectbox(
    "Select City",
    city_list,
    index=city_list.index(default_city)
)

city_df = df[df['City'] == selected_city]

st.subheader(f"EDA for {selected_city}")

# -------------------- Basic Metrics --------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", city_df.shape[0])
col2.metric("Average AQI", round(city_df['AQI'].mean(), 2))
col3.metric("Max AQI", int(city_df['AQI'].max()))


# -------------------- Time Series Analysis --------------------
st.markdown("### 🔹 AQI Trend Over Time")

fig = px.line(
    city_df,
    x="Date",
    y="AQI",
    title="AQI Trend Over Time",
    markers=True
)
st.plotly_chart(fig, use_container_width=True)

# -------------------- AQI Bucket Distribution --------------------
st.markdown("### 🔹 AQI Category Distribution")

bucket_counts = city_df['AQI_Bucket'].value_counts().reset_index()
bucket_counts.columns = ['AQI_Bucket', 'Count']

fig = px.bar(
    bucket_counts,
    x="AQI_Bucket",
    y="Count",
    title="AQI Category Distribution"
)
st.plotly_chart(fig, use_container_width=True)

# -------------------- AQI Distribution --------------------
st.markdown("### 🔹 AQI Distribution")

fig, ax = plt.subplots()
sns.histplot(city_df['AQI'], kde=True, ax=ax)
st.pyplot(fig)

# -------------------- Pollutant Distribution --------------------
st.markdown("### 🔹 Major Pollutants Distribution")

pollutants = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3']
available_pollutants = [p for p in pollutants if p in city_df.columns]

cols = 3
rows = (len(available_pollutants) + cols - 1) // cols

fig = sp.make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles=available_pollutants
)

for i, pol in enumerate(available_pollutants):
    r = i // cols + 1
    c = i % cols + 1

    fig.add_trace(
        px.histogram(
            city_df,
            x=pol,
            nbins=40,
            opacity=0.75,
            color_discrete_sequence=["#1f77b4"],
            marginal="rug"   # gives KDE-like feel
        ).data[0],
        row=r,
        col=c
    )

fig.update_layout(
    height=350 * rows,
    showlegend=False,
    title_text="Distribution of Major Pollutants",
    title_x=0.5,
    template="plotly_white"
)

fig.update_xaxes(title_text="")
fig.update_yaxes(title_text="Count")

st.plotly_chart(fig, use_container_width=True)
# -------------------- AQI vs Pollutants --------------------
st.markdown("### 🔹 Pollutants vs AQI")

cols = 3
rows = (len(available_pollutants) + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
axes = axes.flatten()
for i, pol in enumerate(available_pollutants):
    sns.scatterplot(
        x=pol,
        y='AQI',
        data=city_df,
        ax=axes[i]
    )
    axes[i].set_title(f"{pol} vs AQI")

plt.tight_layout()
st.pyplot(fig)






# -------------------- Correlation Heatmap --------------------
st.markdown("### 🔹 Correlation Heatmap")

corr_cols = available_pollutants + ['AQI']
corr = city_df[corr_cols].corr()

fig = px.imshow(
    corr.round(2),
    text_auto=True,
    aspect="auto"
)
st.plotly_chart(fig, use_container_width=True)


# -------------------- Monthly Trend --------------------
st.markdown("### 🔹 Monthly AQI Trend")

city_df['Month'] = city_df['Date'].dt.month
monthly_aqi = (
    city_df.groupby('Month')['AQI']
    .mean()
    .reset_index()
    .sort_values('Month')
)

fig = px.line(
    monthly_aqi,
    x='Month',
    y='AQI',
    markers=True
)
fig.update_xaxes(dtick=1)
st.plotly_chart(fig, use_container_width=True)
