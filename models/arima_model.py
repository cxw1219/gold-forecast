import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import RobustScaler
import pmdarima as pm
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

class ARIMAPredictor:
    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        self.data = data_dict
        self.model = None
        self.scaler = RobustScaler()
        self.is_differenced = False
        self.d_order = 0
        self.exog_scaler = None
        
        if 'complete' not in self.data:
            raise ValueError("Expected 'complete' key in data dictionary")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data and ensure index alignment"""
        df_clean = df.copy()
        
        # Ensure datetime index
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean.index = pd.to_datetime(df_clean.index)
        
        # Sort index
        df_clean = df_clean.sort_index()
        
        # Replace infinite values with NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill NaN values for numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].ffill().bfill()
            
            # If any NaN still exist, replace with column mean
            if df_clean[col].isna().any():
                mean_val = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(mean_val)
        
        return df_clean

    def _prepare_features(self) -> Tuple[pd.Series, pd.DataFrame]:
        """Prepare aligned features and target"""
        try:
            main_df = self.data['complete'].copy()
            main_df = self._clean_data(main_df)
            
            # Extract target (gold price)
            target = main_df['close']
            
            # Prepare external features
            exog_features = []
            
            # Add USDX if available
            if 'usdx' in main_df.columns:
                exog_features.append('usdx')
                logger.info("Using USDX as external feature")
            
            # Add bond yields if available
            if 'bond_yield' in main_df.columns:
                exog_features.append('bond_yield')
                logger.info("Using bond yield as external feature")
            
            if exog_features:
                exog = main_df[exog_features].copy()
                # Scale features
                self.exog_scaler = RobustScaler()
                exog_scaled = pd.DataFrame(
                    self.exog_scaler.fit_transform(exog),
                    index=exog.index,
                    columns=exog_features
                )
                logger.info(f"Prepared external features: {', '.join(exog_features)}")
                
                # Calculate correlations with gold
                correlations = {}
                for col in exog_features:
                    corr = main_df['close'].corr(main_df[col])
                    correlations[col] = corr
                logger.info("Feature correlations with gold:")
                for feat, corr in correlations.items():
                    logger.info(f"{feat}: {corr:.3f}")
                
                return target, exog_scaled
            else:
                logger.info("No external features available")
                return target, None
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def fit(self):
        """Fit the ARIMA model with external features"""
        try:
            # Prepare features
            target, exog = self._prepare_features()
            
            # Find best parameters
            p, d, q = self._find_best_parameters(target)
            logger.info(f"Selected ARIMA orders: p={p}, d={d}, q={q}")
            
            # Ensure consistent frequency
            index = pd.date_range(start=target.index[0], end=target.index[-1], freq='4H')
            target = target.reindex(index).fillna(method='ffill')
            if exog is not None:
                exog = exog.reindex(index).fillna(method='ffill')
            
            # Fit ARIMA model
            self.model = SARIMAX(
                target,
                exog=exog,
                order=(p, d, q),
                enforce_stationarity=False
            )
            self.fit_result = self.model.fit(disp=False)
            
            # Log model summary
            logger.info("\nModel Summary:")
            logger.info(f"AIC: {self.fit_result.aic:.2f}")
            logger.info(f"BIC: {self.fit_result.bic:.2f}")
            
            if exog is not None:
                logger.info("\nFeature Coefficients:")
                for name, coef in zip(exog.columns, self.fit_result.params[1:]):
                    logger.info(f"{name}: {coef:.4f}")
            
            logger.info("ARIMA model successfully fitted")
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            raise

    def predict(self, steps: int = 2) -> pd.DataFrame:
        """Generate predictions with confidence intervals"""
        try:
            # Get forecast
            if self.exog_scaler is not None:
                # For demonstration, use last known values of external features
                last_exog = self.model.exog[-1:]
                future_exog = np.tile(last_exog, (steps, 1))
                forecast = self.fit_result.get_forecast(steps=steps, exog=future_exog)
            else:
                forecast = self.fit_result.get_forecast(steps=steps)
            
            mean_forecast = forecast.predicted_mean
            conf_int = forecast.conf_int(alpha=0.05)
            
            # Create forecast DataFrame with proper datetime index
            last_date = self.data['complete'].index[-1]
            forecast_dates = pd.date_range(start=last_date, periods=steps + 1, freq='4H')[1:]
            
            forecast_df = pd.DataFrame({
                'forecast': mean_forecast.values,
                'lower_bound': conf_int.iloc[:, 0].values,
                'upper_bound': conf_int.iloc[:, 1].values
            }, index=forecast_dates)
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise

    def _find_best_parameters(self, data: pd.Series) -> Tuple[int, int, int]:
        """Find best ARIMA parameters using auto_arima"""
        try:
            model = pm.auto_arima(
                data,
                start_p=0, start_q=0,
                max_p=5, max_q=5, max_d=2,
                seasonal=False,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            return model.order
        except Exception as e:
            logger.error(f"Error in parameter selection: {str(e)}")
            return (1, 1, 1)

    def _check_stationarity(self, series: pd.Series) -> bool:
        """Check if the series is stationary using ADF test"""
        result = adfuller(series.dropna())
        return result[1] < 0.05

    def get_model_diagnostics(self) -> Dict:
        """Get model diagnostics and performance metrics"""
        if self.fit_result is None:
            raise ValueError("Model has not been fitted yet")
            
        return {
            'aic': self.fit_result.aic,
            'bic': self.fit_result.bic,
            'residual_std': np.sqrt(self.fit_result.scale),
            'params': self.fit_result.params,
            'mse': self.fit_result.mse
        }