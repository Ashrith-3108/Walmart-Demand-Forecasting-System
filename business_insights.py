"""
Business Insights and Analytics for Demand Forecasting
Provides actionable business insights, recommendations, and KPI tracking
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class BusinessInsightsAnalyzer:
    def __init__(self):
        self.insights = {}
        self.recommendations = []
        
    def analyze_store_performance(self, data):
        """Analyze overall store performance and identify patterns"""
        insights = {}
        
        # Store performance metrics
        store_metrics = data.groupby('Store').agg({
            'Weekly_Sales': ['mean', 'std', 'min', 'max', 'sum'],
            'Date': 'count'
        }).round(2)
        
        store_metrics.columns = ['Avg_Sales', 'Sales_Std', 'Min_Sales', 'Max_Sales', 'Total_Sales', 'Weeks_Count']
        store_metrics['Sales_CV'] = (store_metrics['Sales_Std'] / store_metrics['Avg_Sales'] * 100).round(2)
        
        insights['store_metrics'] = store_metrics
        
        # Performance tiers
        store_metrics['Performance_Tier'] = pd.cut(
            store_metrics['Avg_Sales'], 
            bins=3, 
            labels=['Low', 'Medium', 'High']
        )
        
        # Top and bottom performers
        top_stores = store_metrics.nlargest(5, 'Avg_Sales')
        bottom_stores = store_metrics.nsmallest(5, 'Avg_Sales')
        
        insights['top_performers'] = top_stores
        insights['bottom_performers'] = bottom_stores
        
        # Sales trend analysis
        trend_analysis = self._analyze_sales_trends(data)
        insights['trend_analysis'] = trend_analysis
        
        return insights
    
    def _analyze_sales_trends(self, data):
        """Analyze sales trends across stores"""
        trend_data = data.groupby(['Store', 'Date']).agg({'Weekly_Sales': 'sum'}).reset_index()
        trend_data = trend_data.sort_values(['Store', 'Date'])
        
        # Calculate growth rates
        trend_data['Sales_Growth'] = trend_data.groupby('Store')['Weekly_Sales'].pct_change()
        trend_data['Sales_Growth_MA'] = trend_data.groupby('Store')['Sales_Growth'].rolling(4, min_periods=1).mean().reset_index(0, drop=True)
        
        # Overall trend
        overall_trend = trend_data.groupby('Date')['Weekly_Sales'].mean()
        overall_growth = overall_trend.pct_change().mean() * 100
        
        return {
            'overall_growth_rate': overall_growth,
            'trend_data': trend_data,
            'overall_trend': overall_trend
        }
    
    def analyze_seasonality_patterns(self, data):
        """Analyze seasonal patterns in sales"""
        seasonality = {}
        
        # Monthly seasonality
        monthly_sales = data.groupby(data['Date'].dt.month)['Weekly_Sales'].mean()
        seasonality['monthly_pattern'] = monthly_sales
        
        # Quarterly seasonality
        quarterly_sales = data.groupby(data['Date'].dt.quarter)['Weekly_Sales'].mean()
        seasonality['quarterly_pattern'] = quarterly_sales
        
        # Holiday impact
        if 'Holiday_Flag' in data.columns:
            holiday_sales = data[data['Holiday_Flag'] == 1]['Weekly_Sales'].mean()
            non_holiday_sales = data[data['Holiday_Flag'] == 0]['Weekly_Sales'].mean()
            holiday_uplift = ((holiday_sales - non_holiday_sales) / non_holiday_sales) * 100
            
            seasonality['holiday_impact'] = {
                'holiday_sales': holiday_sales,
                'non_holiday_sales': non_holiday_sales,
                'uplift_percentage': holiday_uplift
            }
        
        # Peak and trough months
        peak_month = monthly_sales.idxmax()
        trough_month = monthly_sales.idxmin()
        seasonality['peak_trough'] = {
            'peak_month': peak_month,
            'trough_month': trough_month,
            'seasonality_strength': (monthly_sales.max() - monthly_sales.min()) / monthly_sales.mean() * 100
        }
        
        return seasonality
    
    def analyze_external_factors(self, data):
        """Analyze impact of external factors on sales"""
        external_analysis = {}
        
        # Economic indicators correlation
        economic_cols = ['CPI', 'Unemployment', 'Fuel_Price']
        correlations = {}
        
        for col in economic_cols:
            if col in data.columns:
                correlation = data['Weekly_Sales'].corr(data[col])
                correlations[col] = correlation
        
        external_analysis['economic_correlations'] = correlations
        
        # Temperature impact (if available)
        if 'Temperature' in data.columns:
            temp_analysis = self._analyze_temperature_impact(data)
            external_analysis['temperature_impact'] = temp_analysis
        
        # Fuel price impact
        if 'Fuel_Price' in data.columns:
            fuel_analysis = self._analyze_fuel_price_impact(data)
            external_analysis['fuel_impact'] = fuel_analysis
        
        return external_analysis
    
    def _analyze_temperature_impact(self, data):
        """Analyze temperature impact on sales"""
        # Create temperature bins
        data['Temp_Bin'] = pd.cut(data['Temperature'], bins=5, labels=['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot'])
        temp_sales = data.groupby('Temp_Bin')['Weekly_Sales'].mean()
        
        return {
            'temperature_sales': temp_sales,
            'optimal_temperature': temp_sales.idxmax(),
            'temperature_sensitivity': temp_sales.std() / temp_sales.mean() * 100
        }
    
    def _analyze_fuel_price_impact(self, data):
        """Analyze fuel price impact on sales"""
        # Create fuel price bins
        data['Fuel_Bin'] = pd.cut(data['Fuel_Price'], bins=5, labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
        fuel_sales = data.groupby('Fuel_Bin')['Weekly_Sales'].mean()
        
        # Correlation analysis
        correlation = data['Weekly_Sales'].corr(data['Fuel_Price'])
        
        return {
            'fuel_price_sales': fuel_sales,
            'correlation': correlation,
            'fuel_sensitivity': fuel_sales.std() / fuel_sales.mean() * 100
        }
    
    def generate_forecast_insights(self, forecast_data, historical_data, store_id):
        """Generate insights from forecast results"""
        insights = {}
        
        # Forecast summary statistics
        forecast_summary = {
            'mean_forecast': forecast_data['yhat'].mean(),
            'forecast_std': forecast_data['yhat'].std(),
            'forecast_range': forecast_data['yhat'].max() - forecast_data['yhat'].min(),
            'trend_direction': 'Increasing' if forecast_data['yhat'].iloc[-1] > forecast_data['yhat'].iloc[0] else 'Decreasing'
        }
        
        insights['forecast_summary'] = forecast_summary
        
        # Compare with historical performance
        historical_mean = historical_data['Weekly_Sales'].mean()
        forecast_mean = forecast_data['yhat'].mean()
        performance_change = ((forecast_mean - historical_mean) / historical_mean) * 100
        
        insights['performance_comparison'] = {
            'historical_mean': historical_mean,
            'forecast_mean': forecast_mean,
            'expected_change': performance_change
        }
        
        # Identify forecast periods of interest
        peak_forecast_week = forecast_data.loc[forecast_data['yhat'].idxmax()]
        trough_forecast_week = forecast_data.loc[forecast_data['yhat'].idxmin()]
        
        insights['forecast_peaks'] = {
            'peak_week': peak_forecast_week,
            'trough_week': trough_forecast_week
        }
        
        return insights
    
    def generate_recommendations(self, insights, forecast_insights=None):
        """Generate actionable business recommendations"""
        recommendations = []
        
        # Store performance recommendations
        if 'store_metrics' in insights:
            store_metrics = insights['store_metrics']
            
            # High variability stores
            high_cv_stores = store_metrics[store_metrics['Sales_CV'] > 30]
            if not high_cv_stores.empty:
                recommendations.append({
                    'category': 'Store Performance',
                    'priority': 'High',
                    'recommendation': f"Focus on {len(high_cv_stores)} stores with high sales variability (CV > 30%). Consider implementing demand smoothing strategies.",
                    'affected_stores': high_cv_stores.index.tolist()
                })
            
            # Low performing stores
            low_performers = store_metrics[store_metrics['Performance_Tier'] == 'Low']
            if not low_performers.empty:
                recommendations.append({
                    'category': 'Store Performance',
                    'priority': 'High',
                    'recommendation': f"Develop improvement plans for {len(low_performers)} low-performing stores. Consider location analysis and competitive assessment.",
                    'affected_stores': low_performers.index.tolist()
                })
        
        # Seasonal recommendations
        if 'seasonality_patterns' in insights:
            seasonality = insights['seasonality_patterns']
            
            if 'peak_trough' in seasonality:
                peak_month = seasonality['peak_trough']['peak_month']
                trough_month = seasonality['peak_trough']['trough_month']
                
                recommendations.append({
                    'category': 'Seasonal Planning',
                    'priority': 'Medium',
                    'recommendation': f"Plan inventory and staffing for peak month ({peak_month}) and develop strategies to boost sales during trough month ({trough_month}).",
                    'timing': f"Peak: Month {peak_month}, Trough: Month {trough_month}"
                })
        
        # Holiday recommendations
        if 'seasonality_patterns' in insights and 'holiday_impact' in insights['seasonality_patterns']:
            holiday_impact = insights['seasonality_patterns']['holiday_impact']
            uplift = holiday_impact['uplift_percentage']
            
            if uplift > 10:
                recommendations.append({
                    'category': 'Holiday Strategy',
                    'priority': 'Medium',
                    'recommendation': f"Holidays show {uplift:.1f}% sales uplift. Consider expanding holiday promotions and inventory planning.",
                    'impact': f"{uplift:.1f}% sales increase"
                })
        
        # External factors recommendations
        if 'external_factors' in insights:
            external = insights['external_factors']
            
            if 'economic_correlations' in external:
                correlations = external['economic_correlations']
                
                for factor, corr in correlations.items():
                    if abs(corr) > 0.3:
                        direction = "positive" if corr > 0 else "negative"
                        recommendations.append({
                            'category': 'External Factors',
                            'priority': 'Medium',
                            'recommendation': f"Monitor {factor} closely - shows {direction} correlation ({corr:.2f}) with sales.",
                            'factor': factor,
                            'correlation': corr
                        })
        
        # Forecast-based recommendations
        if forecast_insights:
            performance_change = forecast_insights['performance_comparison']['expected_change']
            
            if performance_change > 5:
                recommendations.append({
                    'category': 'Forecast Planning',
                    'priority': 'High',
                    'recommendation': f"Forecast indicates {performance_change:.1f}% growth. Plan for increased inventory and staffing needs.",
                    'expected_growth': f"{performance_change:.1f}%"
                })
            elif performance_change < -5:
                recommendations.append({
                    'category': 'Forecast Planning',
                    'priority': 'High',
                    'recommendation': f"Forecast indicates {abs(performance_change):.1f}% decline. Review pricing strategy and promotional activities.",
                    'expected_decline': f"{abs(performance_change):.1f}%"
                })
        
        return recommendations
    
    def create_dashboard_visualizations(self, data, insights, forecast_data=None):
        """Create comprehensive dashboard visualizations"""
        figures = {}
        
        # Store performance heatmap
        if 'store_metrics' in insights:
            store_metrics = insights['store_metrics']
            
            # Performance heatmap
            fig_heatmap = px.imshow(
                store_metrics[['Avg_Sales', 'Sales_CV', 'Total_Sales']].T,
                labels=dict(x="Store", y="Metric", color="Value"),
                title="Store Performance Heatmap",
                aspect="auto"
            )
            figures['store_heatmap'] = fig_heatmap
            
            # Top performers chart
            top_stores = insights['top_performers']
            fig_top = px.bar(
                top_stores.reset_index(),
                x='Store',
                y='Avg_Sales',
                title="Top 5 Performing Stores",
                color='Avg_Sales',
                color_continuous_scale='viridis'
            )
            figures['top_performers'] = fig_top
        
        # Seasonal patterns
        if 'seasonality_patterns' in insights:
            seasonality = insights['seasonality_patterns']
            
            # Monthly seasonality
            monthly_data = seasonality['monthly_pattern'].reset_index()
            monthly_data.columns = ['Month', 'Avg_Sales']
            
            fig_monthly = px.line(
                monthly_data,
                x='Month',
                y='Avg_Sales',
                title="Monthly Sales Seasonality",
                markers=True
            )
            figures['monthly_seasonality'] = fig_monthly
        
        # Trend analysis
        if 'trend_analysis' in insights:
            trend_data = insights['trend_analysis']['overall_trend'].reset_index()
            trend_data.columns = ['Date', 'Avg_Sales']
            
            fig_trend = px.line(
                trend_data,
                x='Date',
                y='Avg_Sales',
                title="Overall Sales Trend",
                markers=True
            )
            figures['sales_trend'] = fig_trend
        
        # Forecast visualization
        if forecast_data is not None:
            fig_forecast = px.line(
                forecast_data,
                x='ds',
                y='yhat',
                title="Sales Forecast",
                labels={'ds': 'Date', 'yhat': 'Forecasted Sales'}
            )
            
            # Add confidence intervals if available
            if 'yhat_lower' in forecast_data.columns and 'yhat_upper' in forecast_data.columns:
                fig_forecast.add_trace(
                    go.Scatter(
                        x=forecast_data['ds'],
                        y=forecast_data['yhat_upper'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,100,80,0.2)',
                        showlegend=False
                    )
                )
                fig_forecast.add_trace(
                    go.Scatter(
                        x=forecast_data['ds'],
                        y=forecast_data['yhat_lower'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,100,80,0.2)',
                        showlegend=False
                    )
                )
            
            figures['forecast'] = fig_forecast
        
        return figures
    
    def generate_executive_summary(self, insights, recommendations):
        """Generate executive summary report"""
        summary = f"""
        ========================================
        EXECUTIVE SUMMARY - DEMAND FORECASTING
        ========================================
        
        KEY FINDINGS:
        """
        
        # Add key metrics
        if 'store_metrics' in insights:
            store_metrics = insights['store_metrics']
            total_stores = len(store_metrics)
            avg_sales = store_metrics['Avg_Sales'].mean()
            total_revenue = store_metrics['Total_Sales'].sum()
            
            summary += f"""
        - Total Stores Analyzed: {total_stores}
        - Average Weekly Sales: ${avg_sales:,.2f}
        - Total Revenue: ${total_revenue:,.2f}
        """
        
        # Add trend information
        if 'trend_analysis' in insights:
            growth_rate = insights['trend_analysis']['overall_growth_rate']
            trend_direction = "positive" if growth_rate > 0 else "negative"
            summary += f"""
        - Overall Growth Trend: {trend_direction} ({growth_rate:.2f}% per week)
        """
        
        # Add seasonal insights
        if 'seasonality_patterns' in insights:
            seasonality = insights['seasonality_patterns']
            if 'peak_trough' in seasonality:
                peak_month = seasonality['peak_trough']['peak_month']
                trough_month = seasonality['peak_trough']['trough_month']
                summary += f"""
        - Peak Sales Month: {peak_month}
        - Lowest Sales Month: {trough_month}
        """
        
        # Add recommendations summary
        summary += f"""
        
        RECOMMENDATIONS SUMMARY:
        - High Priority: {len([r for r in recommendations if r['priority'] == 'High'])} items
        - Medium Priority: {len([r for r in recommendations if r['priority'] == 'Medium'])} items
        
        TOP RECOMMENDATIONS:
        """
        
        high_priority_recs = [r for r in recommendations if r['priority'] == 'High']
        for i, rec in enumerate(high_priority_recs[:3], 1):
            summary += f"""
        {i}. {rec['recommendation']}
        """
        
        return summary
    
    def export_insights_report(self, insights, recommendations, filename):
        """Export comprehensive insights report to file"""
        import json
        
        report = {
            'insights': insights,
            'recommendations': recommendations,
            'summary': self.generate_executive_summary(insights, recommendations),
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

# Example usage
if __name__ == "__main__":
    analyzer = BusinessInsightsAnalyzer()
    
    # Example with dummy data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='W')
    stores = np.random.randint(1, 11, 100)
    sales = np.random.normal(10000, 2000, 100)
    
    dummy_data = pd.DataFrame({
        'Store': stores,
        'Date': dates,
        'Weekly_Sales': sales,
        'Holiday_Flag': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        'CPI': np.random.normal(200, 5, 100),
        'Unemployment': np.random.normal(6, 0.5, 100),
        'Fuel_Price': np.random.normal(3, 0.1, 100)
    })
    
    # Analyze data
    insights = analyzer.analyze_store_performance(dummy_data)
    seasonality = analyzer.analyze_seasonality_patterns(dummy_data)
    external = analyzer.analyze_external_factors(dummy_data)
    
    insights['seasonality_patterns'] = seasonality
    insights['external_factors'] = external
    
    # Generate recommendations
    recommendations = analyzer.generate_recommendations(insights)
    
    # Generate summary
    summary = analyzer.generate_executive_summary(insights, recommendations)
    print(summary)
