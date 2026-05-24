"""
Ensemble models combining BiLSTM, Random Forest, and XGBoost with TimeSeriesSplit validation.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb


class EnsembleRegressor:
    """
    Combine BiLSTM, Random Forest, and XGBoost predictions.
    Weights are learned from validation set.
    """
    
    def __init__(self, bilstm_model=None, rf_model=None, xgb_model=None):
        self.bilstm_model = bilstm_model
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.weights = None
    
    def predict(self, X_features, X_lstm=None):
        """
        Make ensemble predictions.
        
        Args:
            X_features: (n_samples, n_features) - for RF and XGBoost
            X_lstm: (n_samples, window, n_features) - for BiLSTM
        
        Returns:
            predictions: (n_samples, 1)
        """
        predictions = []
        weights = self.weights if self.weights is not None else {"bilstm": 0.4, "rf": 0.3, "xgb": 0.3}
        
        if self.bilstm_model is not None and X_lstm is not None:
            lstm_pred = self.bilstm_model.predict(X_lstm, verbose=0).reshape(-1)
            predictions.append(("bilstm", lstm_pred, weights.get("bilstm", 0.4)))
        
        if self.rf_model is not None:
            rf_pred = self.rf_model.predict(X_features).reshape(-1)
            predictions.append(("rf", rf_pred, weights.get("rf", 0.3)))
        
        if self.xgb_model is not None:
            xgb_pred = self.xgb_model.predict(X_features).reshape(-1)
            predictions.append(("xgb", xgb_pred, weights.get("xgb", 0.3)))
        
        # Weighted average
        if not predictions:
            raise ValueError("No models available for prediction")
        
        weighted_sum = sum(pred * weight for _, pred, weight in predictions)
        total_weight = sum(weight for _, _, weight in predictions)
        
        return (weighted_sum / total_weight).reshape(-1, 1)
    
    def set_weights(self, weights):
        """Set ensemble weights manually."""
        self.weights = weights


def build_random_forest_model(n_estimators=200, max_depth=8, min_samples_leaf=5):
    """Build a Random Forest model for ensemble."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )


def build_xgboost_model(n_estimators=100, max_depth=6, learning_rate=0.1):
    """Build an XGBoost model for ensemble."""
    return xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )


class TimeSeriesEnsembleValidator:
    """
    Validate ensemble using TimeSeriesSplit to prevent data leakage.
    """
    
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
    
    def validate_models(self, X_features, X_lstm, y, bilstm_model=None, 
                       rf_params=None, xgb_params=None):
        """
        Validate models using TimeSeriesSplit.
        Returns validation metrics and learned ensemble weights.
        
        Args:
            X_features: (n_samples, n_features)
            X_lstm: (n_samples, window, n_features)
            y: (n_samples, 1)
            bilstm_model: Pre-trained BiLSTM model
            rf_params: dict of RandomForest parameters
            xgb_params: dict of XGBoost parameters
        
        Returns:
            dict with validation_scores, ensemble_weights, prediction_errors
        """
        rf_params = rf_params or {}
        xgb_params = xgb_params or {}
        
        all_bilstm_preds = []
        all_rf_preds = []
        all_xgb_preds = []
        all_y = []
        
        fold_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.tscv.split(X_features)):
            # Split data
            X_train_features = X_features[train_idx]
            X_test_features = X_features[test_idx]
            X_train_lstm = X_lstm[train_idx]
            X_test_lstm = X_lstm[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            # Train RF
            rf = build_random_forest_model(**rf_params)
            rf.fit(X_train_features, y_train.reshape(-1))
            rf_pred = rf.predict(X_test_features).reshape(-1)
            all_rf_preds.extend(rf_pred)
            
            # Train XGBoost
            xgb_model = build_xgboost_model(**xgb_params)
            xgb_model.fit(X_train_features, y_train.reshape(-1))
            xgb_pred = xgb_model.predict(X_test_features).reshape(-1)
            all_xgb_preds.extend(xgb_pred)
            
            # BiLSTM predictions (if model provided)
            if bilstm_model is not None:
                bilstm_pred = bilstm_model.predict(X_test_lstm, verbose=0).reshape(-1)
                all_bilstm_preds.extend(bilstm_pred)
            
            all_y.extend(y_test.reshape(-1))
            
            fold_scores.append({
                "fold": fold_idx,
                "test_size": len(test_idx),
            })
        
        # Compute MSE for each model
        all_y = np.array(all_y)
        
        model_errors = {}
        if all_bilstm_preds:
            bilstm_mse = np.mean((np.array(all_bilstm_preds) - all_y) ** 2)
            model_errors["bilstm"] = bilstm_mse
        
        rf_mse = np.mean((np.array(all_rf_preds) - all_y) ** 2)
        model_errors["rf"] = rf_mse
        
        xgb_mse = np.mean((np.array(all_xgb_preds) - all_y) ** 2)
        model_errors["xgb"] = xgb_mse
        
        # Learn weights inversely proportional to MSE
        total_error = sum(model_errors.values())
        weights = {model: 1 - (err / total_error) for model, err in model_errors.items()}
        # Normalize weights
        weight_sum = sum(weights.values())
        weights = {model: w / weight_sum for model, w in weights.items()}
        
        return {
            "fold_scores": fold_scores,
            "model_errors": model_errors,
            "ensemble_weights": weights,
            "predictions": {
                "bilstm": all_bilstm_preds,
                "rf": all_rf_preds,
                "xgb": all_xgb_preds,
            },
            "actual": all_y,
        }
