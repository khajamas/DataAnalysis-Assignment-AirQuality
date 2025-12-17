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
        st.title("Air Quality Analysis")
        
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
                     





# Time series analysis
st.subheader("Time Series Analysis")
col1, col2 = st.columns(2)
        
with col1:
        time_resolution = st.selectbox("Time Resolution", ["Daily", "Monthly", "Yearly"])
        
with col2:
        city_filter = st.multiselect("Select Cities", df['City'].unique(), default=["Ahmedabad"])

try:
    filtered_df = df[df['City'].isin(city_filter)]
            
    if time_resolution == "Monthly":
        time_df = filtered_df.groupby(['Year', 'Month', 'City'])['AQI'].mean().reset_index()
        time_df['Date'] = pd.to_datetime(time_df[['Year', 'Month']].assign(DAY=1))
        fig = px.line(time_df, x='Date', y='AQI', color='City', 
                             title="Monthly Average AQI Trend")
    elif time_resolution == "Yearly":
        time_df = filtered_df.groupby(['Year', 'City'])['AQI'].mean().reset_index()
        fig = px.line(time_df, x='Year', y='AQI', color='City', 
                             title="Yearly Average AQI Trend")
    else:
        fig = px.line(filtered_df, x='Date', y='AQI', color='City', 
                             title="Daily AQI Trend")
            
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    show_error(f"Error generating time series plot: {str(e)}")


st.subheader("Pollutant Correlation with AQI")
pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
existing_pollutants = [col for col in pollutant_cols if col in df.columns]
        
try:
    selected_pollutants = st.multiselect("Select Pollutants", existing_pollutants, 
                                               default=['PM2.5', 'PM10'] if 'PM2.5' in existing_pollutants else existing_pollutants[:3])
            
    if selected_pollutants:
        corr_df = df[selected_pollutants + ['AQI']].corr()
        fig = px.imshow(corr_df, text_auto=True, aspect="auto", 
                               title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
                
except Exception as e:
    show_error(f"Error generating correlation matrix: {str(e)}")

selected_pollutants = st.multiselect("Select Pollutants", existing_pollutants, 
                                               default=['PM2.5', 'PM10', 'NO2'] if 'PM2.5' in existing_pollutants else existing_pollutants[:3])
            
if selected_pollutants:
                corr_df = df[selected_pollutants + ['AQI']].corr()
                fig = px.imshow(corr_df, text_auto=True, aspect="auto", 
                               title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
                
                # Add pollutant information section
                st.markdown("### Pollutant Information")
                selected_pollutant = st.selectbox("Select a pollutant to learn more", selected_pollutants)
                
                pollutant_info = get_pollutant_impact_info(selected_pollutant)
                st.markdown(
                    f"""
                    <div class='impact-factor'>
                        <h4>{selected_pollutant}</h4>
                        <p><strong>What is it?</strong> {pollutant_info['description']}</p>
                        <p><strong>Main sources:</strong> {pollutant_info['sources']}</p>
                        <p><strong>Health effects:</strong> {pollutant_info['health_effects']}</p>
                        <p><strong>How to reduce:</strong> {pollutant_info['mitigation']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
# Custom 2D/3D visualization
st.subheader("Custom 2D Visualization")
try:       
    cols = df.select_dtypes(include=np.number).columns.tolist()
            
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis", cols, index=cols.index('PM2.5') if 'PM2.5' in cols else 0)
    with col2:
        y_axis = st.selectbox("Y-axis", cols, index=cols.index('AQI') if 'AQI' in cols else 1)
                
    fig = px.scatter(df, x=x_axis, y=y_axis, color='AQI_Bucket' if 'AQI_Bucket' in df.columns else None,color_discrete_map=aqi_color_map,
                                 hover_data=['City', 'Date_Display'],
                                 title=f"{x_axis} vs {y_axis}")
    st.plotly_chart(fig, use_container_width=True)
            
except Exception as e:
    show_error(f"Error in custom visualization: {str(e)}")


#AQI by City
st.subheader('AQI by City')

cities = df['City'].unique()
selected_city = st.selectbox('Select a city', cities)

city_data = df[df['City'] == selected_city].copy()
city_data['Year'] = city_data['Date'].dt.year

yearly_aqi = (
    city_data
    .groupby('Year', as_index=False)['AQI']
    .mean()
)

fig_aqi_city = px.bar(
    yearly_aqi,
    x='Year',
    y='AQI',
    text_auto='.1f',
    color='AQI',
    color_continuous_scale='Reds',
    title=f'AQI Trend for {selected_city}',
    template='plotly_white'
)

fig_aqi_city.update_layout(
    xaxis_title='Year',
    yaxis_title='Average AQI',
    coloraxis_showscale=False
)

st.plotly_chart(fig_aqi_city, use_container_width=True)

st.write('- The graph shows the **average AQI trend** for the selected city over the years.')

#AQI By Year
years = sorted(df['Date'].dt.year.unique())
selected_year = st.selectbox('Select a year', years)

year_data = df[df['Date'].dt.year == selected_year]

city_aqi_year = (
    year_data
    .groupby('City', as_index=False)['AQI']
    .mean()
    .sort_values('AQI', ascending=False)
)

fig_aqi_year = px.bar(
    city_aqi_year,
    x='City',
    y='AQI',
    color='AQI',
    text_auto='.1f',
    color_continuous_scale='OrRd',
    title=f'AQI by City for {selected_year}',
    template='plotly_white'
)

fig_aqi_year.update_layout(
    xaxis_title='City',
    yaxis_title='Average AQI',
    xaxis_tickangle=-45,
    coloraxis_showscale=False
)

st.plotly_chart(fig_aqi_year, use_container_width=True)

st.write('- The graph shows the **average AQI for different cities** in the selected year.')


#Pollutants Over Time
st.subheader('Pollutants Over Time')
pollutants = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
    'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
]

available_pollutants = [p for p in pollutants if p in df.columns]

selected_pollutants = st.multiselect(
    'Select pollutants',
    available_pollutants,
    default=['PM2.5', 'PM10']
)

cities = df['City'].unique()
selected_city = st.selectbox('Select a city', cities, key='pollutant_city')

city_data = df[df['City'] == selected_city]

fig_pollutants = px.line(
    city_data,
    x='Date',
    y=selected_pollutants,
    title=f'Pollutants Over Time for {selected_city}',
    template='plotly_white'
)

fig_pollutants.update_layout(
    xaxis_title='Date',
    yaxis_title='Pollutant Level',
    legend_title='Pollutant',
    hovermode='x unified'
)

st.plotly_chart(fig_pollutants, use_container_width=True)

st.write('- The graph shows the levels of selected pollutants over time for the selected city.')
