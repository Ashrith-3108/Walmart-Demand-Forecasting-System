"""
Advanced Model Evaluation for Demand Forecasting
Provides comprehensive evaluation metrics and validation techniques
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DemandForecastingEvaluator:
    def __init__(self):
        self.metrics = {}
        self.residuals = {}
        self.forecasts = {}
        
    def calculate_metrics(self, y_true, y_pred, model_name="Model"):
        """Calculate comprehensive evaluation metrics"""
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Percentage errors
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
        
        # Directional accuracy
        directional_accuracy = self._calculate_directional_accuracy(y_true, y_pred)
        
        # Store metrics
        self.metrics[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'SMAPE': smape,
            'Directional_Accuracy': directional_accuracy
        }
        
        # Store residuals
        self.residuals[model_name] = y_true - y_pred
        self.forecasts[model_name] = {'actual': y_true, 'predicted': y_pred}
        
        return self.metrics[model_name]
    
    def _calculate_directional_accuracy(self, y_true, y_pred):
        """Calculate directional accuracy (correctly predicting trend direction)"""
        if len(y_true) < 2:
            return 0
        
        actual_direction = np.diff(y_true) > 0
        predicted_direction = np.diff(y_pred) > 0
        
        correct_directions = np.sum(actual_direction == predicted_direction)
        total_directions = len(actual_direction)
        
        return (correct_directions / total_directions) * 100 if total_directions > 0 else 0
    
    def time_series_cross_validation(self, model, X, y, n_splits=5):
        """Perform time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            cv_scores.append(rmse)
        
        return {
            'mean_rmse': np.mean(cv_scores),
            'std_rmse': np.std(cv_scores),
            'cv_scores': cv_scores
        }
    
    def rolling_window_validation(self, model, data, target_col, window_size=12, step_size=4):
        """Perform rolling window validation"""
        predictions = []
        actuals = []
        
        for i in range(window_size, len(data), step_size):
            # Training window
            train_data = data.iloc[i-window_size:i]
            test_data = data.iloc[i:i+step_size]
            
            # Prepare features (excluding target and date columns)
            exclude_cols = ['Date', target_col]
            feature_cols = [col for col in train_data.columns if col not in exclude_cols]
            
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            predictions.extend(y_pred)
            actuals.extend(y_test)
        
        return np.array(actuals), np.array(predictions)
    
    def residual_analysis(self, model_name):
        """Perform comprehensive residual analysis"""
        if model_name not in self.residuals:
            raise ValueError(f"No residuals found for model: {model_name}")
        
        residuals = self.residuals[model_name]
        
        # Basic statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'normality_test_pvalue': stats.normaltest(residuals)[1]
        }
        
        # Autocorrelation
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        residual_stats['autocorrelation'] = autocorr
        
        return residual_stats
    
    def plot_residuals(self, model_name, figsize=(15, 10)):
        """Plot comprehensive residual analysis"""
        if model_name not in self.residuals:
            raise ValueError(f"No residuals found for model: {model_name}")
        
        residuals = self.residuals[model_name]
        forecasts = self.forecasts[model_name]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Residual Analysis for {model_name}', fontsize=16)
        
        # 1. Residuals vs Predicted
        axes[0, 0].scatter(forecasts['predicted'], residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        
        # 2. Residuals histogram
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')
        
        # 3. Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('Q-Q Plot')
        
        # 4. Residuals over time
        axes[1, 0].plot(residuals, alpha=0.7)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals Over Time')
        
        # 5. Actual vs Predicted
        axes[1, 1].scatter(forecasts['actual'], forecasts['predicted'], alpha=0.6)
        min_val = min(forecasts['actual'].min(), forecasts['predicted'].min())
        max_val = max(forecasts['actual'].max(), forecasts['predicted'].max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted')
        
        # 6. Residuals autocorrelation
        pd.Series(residuals).plot(kind='autocorrelation', ax=axes[1, 2])
        axes[1, 2].set_title('Residuals Autocorrelation')
        
        plt.tight_layout()
        return fig
    
    def compare_models(self, model_names):
        """Compare multiple models using their metrics"""
        comparison_data = []
        
        for model_name in model_names:
            if model_name in self.metrics:
                metrics = self.metrics[model_name]
                metrics['Model'] = model_name
                comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('Model')
        
        return comparison_df
    
    def plot_model_comparison(self, model_names, metric='RMSE', figsize=(10, 6)):
        """Plot comparison of models based on a specific metric"""
        comparison_df = self.compare_models(model_names)
        
        if metric not in comparison_df.columns:
            raise ValueError(f"Metric {metric} not found in comparison data")
        
        plt.figure(figsize=figsize)
        comparison_df[metric].plot(kind='bar')
        plt.title(f'Model Comparison - {metric}')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    def forecast_accuracy_decomposition(self, y_true, y_pred):
        """Decompose forecast errors into bias, variance, and covariance"""
        # Calculate components
        bias = np.mean(y_pred - y_true)
        variance = np.var(y_pred)
        actual_variance = np.var(y_true)
        covariance = np.cov(y_true, y_pred)[0, 1]
        
        # Calculate MSE components
        mse = mean_squared_error(y_true, y_pred)
        bias_squared = bias ** 2
        variance_component = variance + actual_variance - 2 * covariance
        
        decomposition = {
            'MSE': mse,
            'Bias_Squared': bias_squared,
            'Variance_Component': variance_component,
            'Bias_Percentage': (bias_squared / mse) * 100,
            'Variance_Percentage': (variance_component / mse) * 100
        }
        
        return decomposition
    
    def calculate_confidence_intervals(self, y_true, y_pred, confidence_level=0.95):
        """Calculate prediction confidence intervals"""
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        
        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * residual_std
        
        confidence_intervals = {
            'lower_bound': y_pred - margin_of_error,
            'upper_bound': y_pred + margin_of_error,
            'confidence_level': confidence_level,
            'coverage_rate': np.mean((y_true >= y_pred - margin_of_error) & 
                                   (y_true <= y_pred + margin_of_error))
        }
        
        return confidence_intervals
    
    def generate_evaluation_report(self, model_name):
        """Generate a comprehensive evaluation report"""
        if model_name not in self.metrics:
            raise ValueError(f"No metrics found for model: {model_name}")
        
        metrics = self.metrics[model_name]
        residual_stats = self.residual_analysis(model_name)
        
        report = f"""
        ========================================
        EVALUATION REPORT FOR {model_name.upper()}
        ========================================
        
        PERFORMANCE METRICS:
        - RMSE: {metrics['RMSE']:.2f}
        - MAE: {metrics['MAE']:.2f}
        - R²: {metrics['R2']:.4f}
        - MAPE: {metrics['MAPE']:.2f}%
        - SMAPE: {metrics['SMAPE']:.2f}%
        - Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%
        
        RESIDUAL ANALYSIS:
        - Mean: {residual_stats['mean']:.4f}
        - Standard Deviation: {residual_stats['std']:.4f}
        - Skewness: {residual_stats['skewness']:.4f}
        - Kurtosis: {residual_stats['kurtosis']:.4f}
        - Autocorrelation: {residual_stats['autocorrelation']:.4f}
        - Normality Test p-value: {residual_stats['normality_test_pvalue']:.4f}
        
        INTERPRETATION:
        """
        
        # Add interpretation
        if metrics['R2'] > 0.8:
            report += "- Excellent model fit (R² > 0.8)\n"
        elif metrics['R2'] > 0.6:
            report += "- Good model fit (R² > 0.6)\n"
        else:
            report += "- Poor model fit (R² < 0.6)\n"
        
        if abs(residual_stats['mean']) < 0.1:
            report += "- Low bias in predictions\n"
        else:
            report += "- Significant bias detected\n"
        
        if abs(residual_stats['autocorrelation']) < 0.1:
            report += "- Low residual autocorrelation\n"
        else:
            report += "- High residual autocorrelation detected\n"
        
        if residual_stats['normality_test_pvalue'] > 0.05:
            report += "- Residuals appear normally distributed\n"
        else:
            report += "- Residuals are not normally distributed\n"
        
        return report
    
    def save_evaluation_results(self, filename):
        """Save evaluation results to file"""
        results = {
            'metrics': self.metrics,
            'residuals': {k: v.tolist() for k, v in self.residuals.items()},
            'forecasts': {k: {'actual': v['actual'].tolist(), 'predicted': v['predicted'].tolist()} 
                         for k, v in self.forecasts.items()}
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

# Example usage
if __name__ == "__main__":
    evaluator = DemandForecastingEvaluator()
    
    # Example with dummy data
    np.random.seed(42)
    y_true = np.random.normal(100, 10, 100)
    y_pred = y_true + np.random.normal(0, 5, 100)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, "Example_Model")
    print("Metrics:", metrics)
    
    # Generate report
    report = evaluator.generate_evaluation_report("Example_Model")
    print(report)
