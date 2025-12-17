
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        st.title("Data Overview")
        
        # Load data with error handling
        try:
            df = load_data()
        except Exception as e:
            show_error(f"Data loading failed: {str(e)}")
            st.stop()
        

        
        
        # Overview section
        
        st.write(f"Dataset contains {len(df)} records from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")

        # Show raw data
        st.subheader("Raw Dataset Preview")
        try:
                # Create a copy of the dataframe for display purposes
            display_df = df.copy()
                
                # Convert datetime to string for display
            if 'Date' in display_df.columns:
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                
                # Drop the temporary display column if it exists
            if 'Date_Display' in display_df.columns:
                display_df = display_df.drop(columns=['Date_Display'])
                
            st.dataframe(display_df, use_container_width=True)
        except Exception as e:
            show_error(f"Error displaying data: {str(e)}")
                
        # Basic statistics
        st.subheader("Basic Statistical Summary")
        try:
            st.write(df.describe())
        except Exception as e:
            show_error(f"Error generating statistics: {str(e)}")

        # Visualization section


      
except Exception as e:
        show_error(f"Application Error: {str(e)}")
        st.error(traceback.format_exc())



