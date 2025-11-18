"""
Advanced Data Preprocessing for Demand Forecasting
Handles feature engineering, seasonality detection, and data validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class DemandForecastingPreprocessor:
    def __init__(self, country='US'):
        self.country = country
        self.us_holidays = holidays.US()
        self.scaler = StandardScaler()
        self.feature_selector = None
        
    def load_and_validate_data(self, file_path):
        """Load and validate the input dataset"""
        try:
            data = pd.read_csv(file_path)
            data.columns = [col.strip() for col in data.columns]
            
            # Validate required columns
            required_cols = {'Store', 'Date', 'Weekly_Sales'}
            missing_cols = required_cols - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Data type validation
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data['Weekly_Sales'] = pd.to_numeric(data['Weekly_Sales'], errors='coerce')
            data['Store'] = data['Store'].astype(int)
            
            # Remove invalid data
            data = data.dropna(subset=['Date', 'Weekly_Sales'])
            
            # Sort by store and date
            data = data.sort_values(['Store', 'Date']).reset_index(drop=True)
            
            return data
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def load_and_validate_data_from_df(self, data):
        """Load and validate data from existing DataFrame"""
        try:
            data = data.copy()
            data.columns = [col.strip() for col in data.columns]
            
            # Validate required columns
            required_cols = {'Store', 'Date', 'Weekly_Sales'}
            missing_cols = required_cols - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Data type validation
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data['Weekly_Sales'] = pd.to_numeric(data['Weekly_Sales'], errors='coerce')
            data['Store'] = data['Store'].astype(int)
            
            # Remove invalid data
            data = data.dropna(subset=['Date', 'Weekly_Sales'])
            
            # Sort by store and date
            data = data.sort_values(['Store', 'Date']).reset_index(drop=True)
            
            return data
            
        except Exception as e:
            raise Exception(f"Error validating data: {str(e)}")
    
    def engineer_time_features(self, data):
        """Create comprehensive time-based features"""
        data = data.copy()
        
        # Basic time features
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Week'] = data['Date'].dt.isocalendar().week
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['Quarter'] = data['Date'].dt.quarter
        
        # Cyclical encoding for seasonality
        data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
        data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)
        data['Week_Sin'] = np.sin(2 * np.pi * data['Week'] / 52)
        data['Week_Cos'] = np.cos(2 * np.pi * data['Week'] / 52)
        data['DayOfWeek_Sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
        data['DayOfWeek_Cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
        
        # Holiday features
        data['Is_Holiday'] = data['Date'].apply(lambda x: x in self.us_holidays)
        data['Days_To_Holiday'] = data['Date'].apply(self._days_to_next_holiday)
        data['Days_From_Holiday'] = data['Date'].apply(self._days_from_last_holiday)
        
        # Special retail periods
        data['Is_BlackFriday'] = self._is_black_friday(data['Date'])
        data['Is_Christmas'] = self._is_christmas_period(data['Date'])
        data['Is_Summer'] = data['Month'].isin([6, 7, 8])
        data['Is_Winter'] = data['Month'].isin([12, 1, 2])
        
        return data
    
    def _days_to_next_holiday(self, date):
        """Calculate days to next major holiday"""
        try:
            # Get all holiday dates for the next few years
            current_year = date.year
            holiday_dates = []
            
            # Check current year and next year for holidays
            for year in [current_year, current_year + 1]:
                year_holidays = self.us_holidays.get_list(f"{year}-01-01", f"{year}-12-31")
                holiday_dates.extend(year_holidays)
            
            # Find next holiday after the given date
            future_holidays = [d for d in holiday_dates if d > date]
            if future_holidays:
                next_holiday = min(future_holidays)
                return (next_holiday - date).days
            return 365  # Default to 1 year if no holiday found
        except:
            return 365  # Return default if any error occurs
    
    def _days_from_last_holiday(self, date):
        """Calculate days from last major holiday"""
        try:
            # Get all holiday dates for the past few years
            current_year = date.year
            holiday_dates = []
            
            # Check current year and previous year for holidays
            for year in [current_year - 1, current_year]:
                year_holidays = self.us_holidays.get_list(f"{year}-01-01", f"{year}-12-31")
                holiday_dates.extend(year_holidays)
            
            # Find last holiday before the given date
            past_holidays = [d for d in holiday_dates if d < date]
            if past_holidays:
                last_holiday = max(past_holidays)
                return (date - last_holiday).days
            return 365  # Default to 1 year if no holiday found
        except:
            return 365  # Return default if any error occurs
    
    def _is_black_friday(self, dates):
        """Identify Black Friday period (day after Thanksgiving)"""
        black_friday_dates = []
        for date in dates:
            try:
                # Get Thanksgiving date for the same year
                year = date.year
                thanksgiving = None
                
                # Get all holidays for the year
                year_holidays = self.us_holidays.get_list(f"{year}-01-01", f"{year}-12-31")
                
                # Find Thanksgiving
                for holiday_date in year_holidays:
                    holiday_name = self.us_holidays.get(holiday_date)
                    if holiday_name and 'Thanksgiving' in holiday_name:
                        thanksgiving = holiday_date
                        break
                
                if thanksgiving:
                    black_friday = thanksgiving + timedelta(days=1)
                    black_friday_dates.append(date == black_friday)
                else:
                    black_friday_dates.append(False)
            except:
                black_friday_dates.append(False)
        return black_friday_dates
    
    def _is_christmas_period(self, dates):
        """Identify Christmas shopping period (December 1-25)"""
        return (dates.dt.month == 12) & (dates.dt.day <= 25)
    
    def create_lag_features(self, data, lags=[1, 2, 3, 4, 8, 12]):
        """Create lag features for time series forecasting"""
        data = data.copy()
        
        for store in data['Store'].unique():
            store_data = data[data['Store'] == store].copy()
            store_data = store_data.sort_values('Date')
            
            for lag in lags:
                data.loc[data['Store'] == store, f'Lag_{lag}'] = store_data['Weekly_Sales'].shift(lag)
        
        # Rolling statistics
        for store in data['Store'].unique():
            store_data = data[data['Store'] == store].copy()
            store_data = store_data.sort_values('Date')
            
            # Rolling means
            for window in [4, 8, 12]:
                data.loc[data['Store'] == store, f'Rolling_Mean_{window}'] = store_data['Weekly_Sales'].rolling(window=window, min_periods=1).mean()
                data.loc[data['Store'] == store, f'Rolling_Std_{window}'] = store_data['Weekly_Sales'].rolling(window=window, min_periods=1).std()
        
        return data
    
    def engineer_external_features(self, data):
        """Create external features and interactions"""
        data = data.copy()
        
        # Economic indicators (if available)
        if 'CPI' in data.columns:
            data['CPI_Change'] = data.groupby('Store')['CPI'].pct_change()
            data['CPI_Rolling_Mean'] = data.groupby('Store')['CPI'].rolling(4, min_periods=1).mean().reset_index(0, drop=True)
        
        if 'Unemployment' in data.columns:
            data['Unemployment_Change'] = data.groupby('Store')['Unemployment'].pct_change()
            data['Unemployment_Rolling_Mean'] = data.groupby('Store')['Unemployment'].rolling(4, min_periods=1).mean().reset_index(0, drop=True)
        
        if 'Fuel_Price' in data.columns:
            data['Fuel_Price_Change'] = data.groupby('Store')['Fuel_Price'].pct_change()
            data['Fuel_Price_Rolling_Mean'] = data.groupby('Store')['Fuel_Price'].rolling(4, min_periods=1).mean().reset_index(0, drop=True)
        
        # Interaction features
        if all(col in data.columns for col in ['CPI', 'Unemployment']):
            data['CPI_Unemployment_Interaction'] = data['CPI'] * data['Unemployment']
        
        if all(col in data.columns for col in ['Fuel_Price', 'CPI']):
            data['Fuel_CPI_Interaction'] = data['Fuel_Price'] * data['CPI']
        
        return data
    
    def detect_seasonality(self, data, store_id):
        """Detect seasonality patterns in the data"""
        store_data = data[data['Store'] == store_id].sort_values('Date')
        
        # Weekly seasonality
        weekly_pattern = store_data.groupby('DayOfWeek')['Weekly_Sales'].mean()
        weekly_strength = weekly_pattern.std() / weekly_pattern.mean()
        
        # Monthly seasonality
        monthly_pattern = store_data.groupby('Month')['Weekly_Sales'].mean()
        monthly_strength = monthly_pattern.std() / monthly_pattern.mean()
        
        # Quarterly seasonality
        quarterly_pattern = store_data.groupby('Quarter')['Weekly_Sales'].mean()
        quarterly_strength = quarterly_pattern.std() / quarterly_pattern.mean()
        
        return {
            'weekly_strength': weekly_strength,
            'monthly_strength': monthly_strength,
            'quarterly_strength': quarterly_strength,
            'weekly_pattern': weekly_pattern,
            'monthly_pattern': monthly_pattern,
            'quarterly_pattern': quarterly_pattern
        }
    
    def handle_missing_values(self, data, strategy='interpolate'):
        """Handle missing values in the dataset"""
        data = data.copy()
        
        if strategy == 'interpolate':
            # Forward fill for categorical features
            categorical_cols = ['Store', 'Holiday_Flag', 'Is_Holiday', 'Is_BlackFriday', 'Is_Christmas', 'Is_Summer', 'Is_Winter']
            for col in categorical_cols:
                if col in data.columns:
                    data[col] = data[col].fillna(method='ffill')
            
            # Interpolate for numerical features
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if col in data.columns:
                    data[col] = data.groupby('Store')[col].transform(lambda x: x.interpolate(method='linear'))
                    data[col] = data.groupby('Store')[col].transform(lambda x: x.fillna(x.mean()))
        
        elif strategy == 'drop':
            data = data.dropna()
        
        return data
    
    def scale_features(self, data, features_to_scale=None):
        """Scale numerical features"""
        data = data.copy()
        
        if features_to_scale is None:
            # Auto-detect numerical features to scale
            exclude_cols = ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Is_Holiday', 
                           'Is_BlackFriday', 'Is_Christmas', 'Is_Summer', 'Is_Winter']
            features_to_scale = [col for col in data.select_dtypes(include=[np.number]).columns 
                               if col not in exclude_cols]
        
        # Scale features
        data[features_to_scale] = self.scaler.fit_transform(data[features_to_scale])
        
        return data
    
    def select_features(self, data, target='Weekly_Sales', k=20):
        """Select most important features using statistical tests"""
        data = data.copy()
        
        # Prepare features and target
        exclude_cols = ['Store', 'Date', 'Weekly_Sales']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols].fillna(0)
        y = data[target]
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        return selected_features, self.feature_selector.scores_
    
    def prepare_store_data(self, data, store_id, target='Weekly_Sales'):
        """Prepare data for a specific store"""
        store_data = data[data['Store'] == store_id].copy()
        store_data = store_data.sort_values('Date').reset_index(drop=True)
        
        # Remove target column for features
        exclude_cols = ['Store', 'Date', target]
        feature_cols = [col for col in store_data.columns if col not in exclude_cols]
        
        X = store_data[feature_cols]
        y = store_data[target]
        
        return X, y, store_data
    
    def get_feature_importance_ranking(self, feature_names, scores):
        """Get feature importance ranking"""
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Score': scores
        }).sort_values('Score', ascending=False)
        
        return importance_df

# Example usage
if __name__ == "__main__":
    preprocessor = DemandForecastingPreprocessor()
    
    # Load and preprocess data
    # data = preprocessor.load_and_validate_data('walmart_sales.csv')
    # data = preprocessor.engineer_time_features(data)
    # data = preprocessor.create_lag_features(data)
    # data = preprocessor.engineer_external_features(data)
    # data = preprocessor.handle_missing_values(data)
    
    print("Data preprocessing module ready!")
