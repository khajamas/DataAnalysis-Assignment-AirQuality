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

import joblib
import streamlit as st



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


def show_error(message):
    """Display error message in a consistent location"""
    error_placeholder.error(f" {str(message)}")

def display_prediction(prediction, r2_score, importance_df, input_values):
    """Display prediction with appropriate styling and improvement suggestions"""
    aqi_info = get_aqi_category_info(prediction)
    
    if prediction <= 50:
        category = "Good"
        css_class = "good"
    elif prediction <= 100:
        category = "Satisfactory"
        css_class = "satisfactory"
    elif prediction <= 200:
        category = "Moderate"
        css_class = "moderate"
    elif prediction <= 300:
        category = "Poor"
        css_class = "poor"
    elif prediction <= 400:
        category = "Very Poor"
        css_class = "very-poor"
    else:
        category = "Severe"
        css_class = "severe"
    
    st.markdown(
        f"""
        <div class='prediction-box {css_class}'>
            <h2>Predicted AQI: {prediction:.2f}</h2>
            <h3>{category}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Show confidence indicator
    st.write(f"Model Confidence (R² score): {r2_score:.1%}")
    st.progress(float(r2_score))
    
    # Add AQI category information
    st.markdown(f"### Air Quality Assessment")
    st.markdown(f"**Description**: {aqi_info['description']}")
    st.markdown(f"**Health Implications**: {aqi_info['health_implications']}")
    
    # Add recommendations based on AQI category
    st.markdown(
        f"""
        <div class='suggestion-box'>
            <h4>Recommendations</h4>
            <p>{aqi_info['recommendations']}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Show top factors affecting the prediction
    st.subheader("Key Factors Affecting Air Quality")
    
    # Get top 3 influential factors
    top_factors = importance_df.head(3)['Feature'].tolist()
    
    for factor in top_factors:
        factor_info = get_pollutant_impact_info(factor)
        factor_value = input_values[factor]
        
        st.markdown(
            f"""
            <div class='impact-factor'>
                <h4>{factor} ({factor_value:.2f})</h4>
                <p><strong>What is it?</strong> {factor_info['description']}</p>
                <p><strong>Main sources:</strong> {factor_info['sources']}</p>
                <p><strong>Health effects:</strong> {factor_info['health_effects']}</p>
                <p><strong>How to reduce:</strong> {factor_info['mitigation']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Community-level improvement suggestions
    st.subheader("Community-Level Improvement Strategies")
    
    if prediction > 100:  # Only show for moderate and worse AQI
        st.markdown("""
        <div class='suggestion-box'>
            <h4>Short-term Actions</h4>
            <ul>
                <li>Implement odd-even vehicle schemes during high pollution periods</li>
                <li>Temporarily restrict construction activities</li>
                <li>Increase public transport frequency</li>
                <li>Issue public health advisories</li>
                <li>Set up emergency air quality monitoring</li>
            </ul>
        </div>
        <div class='suggestion-box'>
            <h4>Long-term Solutions</h4>
            <ul>
                <li>Promote electric vehicles and establish charging infrastructure</li>
                <li>Expand green spaces and urban forests</li>
                <li>Invest in renewable energy sources</li>
                <li>Implement stricter industrial emission standards</li>
                <li>Develop better waste management systems to prevent open burning</li>
                <li>Improve public transportation networks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def get_pollutant_impact_info(feature_name):
    """Return information about a pollutant's impact and mitigation strategies"""
    pollutant_info = {
        'PM2.5': {
            'description': 'Fine particulate matter less than 2.5 micrometers in diameter that can penetrate deep into the lungs.',
            'sources': 'Vehicle exhaust, industrial emissions, indoor cooking, dust, and wildfires.',
            'health_effects': 'Can cause respiratory problems, decreased lung function, aggravated asthma, and heart disease.',
            'mitigation': 'Use air purifiers with HEPA filters, avoid outdoor activities during high pollution days, proper ventilation when cooking, and reduce use of wood-burning stoves.'
        },
        'PM10': {
            'description': 'Coarse particulate matter between 2.5 and 10 micrometers in diameter.',
            'sources': 'Road dust, construction sites, industrial processes, and agricultural operations.',
            'health_effects': 'Can cause irritation of the eyes, nose, and throat, as well as respiratory issues.',
            'mitigation': 'Reduce dust sources, use dust masks when in construction areas, and keep windows closed during high dust conditions.'
        },
        'NO': {
            'description': 'Nitric oxide, a reactive air pollutant.',
            'sources': 'Combustion processes, particularly in vehicles and power plants.',
            'health_effects': 'Contributes to respiratory problems and can convert to NO2 in the atmosphere.',
            'mitigation': 'Use of catalytic converters, improved combustion technology, and reducing vehicle emissions.'
        },
        'NO2': {
            'description': 'Nitrogen dioxide, a highly reactive gas with a pungent odor.',
            'sources': 'Vehicle exhaust, power plants, and industrial processes.',
            'health_effects': 'Can cause respiratory inflammation, reduced lung function, and increased sensitivity to respiratory infections.',
            'mitigation': 'Promote public transportation, use electric vehicles, and improve industrial emission controls.'
        },
        'NOx': {
            'description': 'A collective term for nitrogen oxides (NO and NO2).',
            'sources': 'Combustion processes in vehicles, power plants, and industrial operations.',
            'health_effects': 'Contributes to respiratory problems, formation of smog, and acid rain.',
            'mitigation': 'Use low-NOx burners, selective catalytic reduction systems, and promote cleaner transportation.'
        },
        'NH3': {
            'description': 'Ammonia, a colorless gas with a pungent odor.',
            'sources': 'Agricultural activities, livestock waste, and fertilizer application.',
            'health_effects': 'Can cause respiratory irritation and contribute to secondary particle formation.',
            'mitigation': 'Improve manure management, use covered manure storage, and implement precise fertilizer application.'
        },
        'CO': {
            'description': 'Carbon monoxide, a colorless, odorless toxic gas.',
            'sources': 'Incomplete combustion in vehicles, residential heating, and industrial processes.',
            'health_effects': 'Reduces oxygen delivery to body organs, causing headaches, dizziness, and at high concentrations, death.',
            'mitigation': 'Regular vehicle maintenance, proper ventilation of combustion sources, and use of carbon monoxide detectors.'
        },
        'SO2': {
            'description': 'Sulfur dioxide, a colorless gas with a strong odor.',
            'sources': 'Burning of fossil fuels containing sulfur, particularly in power plants and industrial processes.',
            'health_effects': 'Can cause respiratory problems, worsen asthma, and contribute to acid rain.',
            'mitigation': 'Use low-sulfur fuels, flue gas desulfurization systems, and promote renewable energy sources.'
        },
        'O3': {
            'description': 'Ground-level ozone, a major component of smog.',
            'sources': 'Formed by chemical reactions between NOx and VOCs in the presence of sunlight.',
            'health_effects': 'Can trigger asthma attacks, cause throat irritation, and reduce lung function.',
            'mitigation': 'Reduce VOC emissions, limit outdoor activities during high ozone days, and improve industrial emission controls.'
        }
    }
    
    return pollutant_info.get(feature_name, {
        'description': 'Information not available for this pollutant.',
        'sources': 'Various industrial and natural sources.',
        'health_effects': 'May cause respiratory and other health issues.',
        'mitigation': 'Follow local air quality guidelines and reduce exposure.'
    })

def get_aqi_category_info(aqi_value):
    """Return information and recommendations based on AQI category"""
    if aqi_value <= 50:
        return {
            'category': 'Good',
            'description': 'Air quality is considered satisfactory, and air pollution poses little or no risk.',
            'recommendations': 'It\'s a great day to be active outside. Enjoy outdoor activities and open windows for fresh air.',
            'health_implications': 'No health implications for the general population.'
        }
    elif aqi_value <= 100:
        return {
            'category': 'Satisfactory',
            'description': 'Air quality is acceptable; however, there may be a moderate health concern for a very small number of people.',
            'recommendations': 'Unusually sensitive people should consider reducing prolonged or heavy exertion.',
            'health_implications': 'May cause minor breathing discomfort to sensitive people.'
        }
    elif aqi_value <= 200:
        return {
            'category': 'Moderate',
            'description': 'Members of sensitive groups may experience health effects.',
            'recommendations': 'People with respiratory or heart disease, the elderly and children should limit prolonged exertion.',
            'health_implications': 'May cause breathing discomfort to people with lung disease, children, and older adults.'
        }
    elif aqi_value <= 300:
        return {
            'category': 'Poor',
            'description': 'Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.',
            'recommendations': 'Active children and adults, and people with respiratory disease should avoid prolonged outdoor exertion.',
            'health_implications': 'May cause respiratory illness on prolonged exposure. Heart disease patients, elderly, and children are at higher risk.'
        }
    elif aqi_value <= 400:
        return {
            'category': 'Very Poor',
            'description': 'Health warnings of emergency conditions. The entire population is more likely to be affected.',
            'recommendations': 'Everyone should avoid outdoor activities. Keep windows and doors closed.',
            'health_implications': 'May cause respiratory impact even on healthy people, and serious health impacts on people with lung/heart disease.'
        }
    else:
        return {
            'category': 'Severe',
            'description': 'Health alert: everyone may experience more serious health effects.',
            'recommendations': 'Everyone should avoid all outdoor exertion. Consider wearing N95 masks outdoors. Use air purifiers indoors.',
            'health_implications': 'May cause respiratory effects even on healthy people. Serious aggravation of heart/lung disease.'
        }


try:
        st.title("Air Prediction")
        
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
                     
feature_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3']
existing_features = [col for col in feature_cols if col in df.columns]       
           



import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Predictive modeling section

        
try:
            # Model training
            st.subheader("Train Prediction Model")
            
            feature_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3']
            existing_features = [col for col in feature_cols if col in df.columns]
            target_col = 'AQI'
            
            if not existing_features:
                raise ValueError("No valid features available for modeling")
            
            X = df[existing_features]
            y = df[target_col]
            
            if 'City' in existing_features:
                le = LabelEncoder()
                X['City'] = le.fit_transform(X['City'])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # model = DecisionTreeRegressor(max_depth=5, random_state=42)
            model = RandomForestRegressor(
                n_estimators=200,   # More trees → better performance (but slower)
                max_depth=10,       # Deeper trees → more complex patterns
                min_samples_split=5,  # Prevent overfitting
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            if hasattr(model, 'feature_names_in_'):
                model.feature_names_in_ = np.array(X_train.columns.tolist())
            
            # Evaluation metrics
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R-squared Score", f"{r2:.3f}")
                st.metric("Mean Absolute Error", f"{mae:.2f}")
            with col2:
                st.metric("Root Mean Squared Error", f"{rmse:.2f}")
                st.metric("Mean Absolute % Error", f"{mape:.2%}")
            with col3:
                st.metric("Mean Squared Error", f"{mse:.2f}")

            # Feature importance
            st.subheader("Feature That Affect AQI the Most")
            # importance_df = pd.DataFrame({
            #     'Feature': existing_features,
            #     'Importance': model.feature_importances_
            # }).sort_values('Importance', ascending=False)
            
            importance_df = pd.DataFrame({
                'Feature': existing_features,
                'Importance': model.feature_importances_  # Same as Decision Tree
            }).sort_values('Importance', ascending=False)

            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                         title="Features That Affect AQI the Most")
            st.plotly_chart(fig, use_container_width=True)
            
            # # Plot the first tree in the forest
            # plt.figure(figsize=(20, 10))
            # plot_tree(model.estimators_[0], feature_names=existing_features, filled=True)
            # plt.show()
            
            # Prediction interface
            st.subheader("Make a Prediction")
            st.write("Enter pollutant levels to predict AQI:")
            
            col1, col2, col3, col4 = st.columns(4)
            inputs = {}
            
            with col1:
                inputs['PM2.5'] = st.number_input("PM2.5", min_value=0.0, value=50.0)
                inputs['PM10'] = st.number_input("PM10", min_value=0.0, value=60.0)
            
            with col2:
                inputs['NO'] = st.number_input("NO", min_value=0.0, value=10.0)
                inputs['NO2'] = st.number_input("NO2", min_value=0.0, value=20.0)
            
            with col3:
                inputs['NOx'] = st.number_input("NOx", min_value=0.0, value=25.0)
                inputs['CO'] = st.number_input("CO", min_value=0.0, value=1.0)
            
            with col4:
                inputs['SO2'] = st.number_input("SO2", min_value=0.0, value=15.0)
                inputs['O3'] = st.number_input("O3", min_value=0.0, value=40.0)
            
            if st.button("Predict AQI"):
                try:
                    input_data = pd.DataFrame([inputs], columns=existing_features)
                    prediction = model.predict(input_data)[0]
                    display_prediction(prediction, r2, importance_df, inputs)
                    
                except Exception as e:
                    show_error(f"Prediction failed: {str(e)}")
                    st.error("Please check your input values and try again.")



except Exception as e:
            show_error(f"Error in predictive modeling section: {str(e)}")
            st.error(traceback.format_exc())
            
            
            