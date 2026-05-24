"""
Lightweight Optuna-based hyperparameter tuning for ensemble models.
"""

import optuna
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from modules.ensemble_models import build_random_forest_model, build_xgboost_model
from modules.data_pipeline import inverse_target


class EnsembleHyperparameterTuner:
    """
    Tune hyperparameters for RF and XGBoost using Optuna with TimeSeriesSplit.
    Lightweight: only 10-20 trials.
    """
    
    def __init__(self, n_trials=10, n_splits=3):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.best_params = None
        self.study = None
    
    def objective(self, trial, X_features, y, target_scaler):
        """
        Objective function for Optuna.
        Uses TimeSeriesSplit for validation.
        """
        # Suggest hyperparameters
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 50, 300, step=50)
        rf_max_depth = trial.suggest_int("rf_max_depth", 4, 12)
        rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 2, 10)
        
        xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 200, step=50)
        xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 9)
        xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True)
        
        # Cross-validate with TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_scores = []
        
        for train_idx, test_idx in tscv.split(X_features):
            X_train = X_features[train_idx]
            X_test = X_features[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            # Train RF
            rf = build_random_forest_model(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_leaf=rf_min_samples_leaf,
            )
            rf.fit(X_train, y_train.reshape(-1))
            rf_pred = rf.predict(X_test).reshape(-1, 1)
            
            # Train XGBoost
            xgb_model = build_xgboost_model(
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=xgb_learning_rate,
            )
            xgb_model.fit(X_train, y_train.reshape(-1))
            xgb_pred = xgb_model.predict(X_test).reshape(-1, 1)
            
            # Ensemble prediction
            ensemble_pred = 0.5 * rf_pred + 0.5 * xgb_pred
            
            # Compute MSE
            mse = np.mean((ensemble_pred - y_test) ** 2)
            fold_scores.append(mse)
        
        return np.mean(fold_scores)
    
    def tune(self, X_features, y, target_scaler, verbose=True):
        """
        Run hyperparameter tuning.
        
        Args:
            X_features: (n_samples, n_features)
            y: (n_samples, 1)
            target_scaler: For reference (not used in tuning)
            verbose: Print progress
        
        Returns:
            dict with best hyperparameters
        """
        # Create study
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        
        # Optimize
        self.study.optimize(
            lambda trial: self.objective(trial, X_features, y, target_scaler),
            n_trials=self.n_trials,
            show_progress_bar=verbose,
        )
        
        self.best_params = self.study.best_params
        
        if verbose:
            print(f"Best MSE: {self.study.best_value:.6f}")
            print(f"Best params: {self.best_params}")
        
        return self.best_params
    
    def get_best_params(self):
        """Get best parameters in format for ensemble models."""
        if self.best_params is None:
            raise ValueError("Tuning not completed. Call tune() first.")
        
        return {
            "rf_params": {
                "n_estimators": self.best_params["rf_n_estimators"],
                "max_depth": self.best_params["rf_max_depth"],
                "min_samples_leaf": self.best_params["rf_min_samples_leaf"],
            },
            "xgb_params": {
                "n_estimators": self.best_params["xgb_n_estimators"],
                "max_depth": self.best_params["xgb_max_depth"],
                "learning_rate": self.best_params["xgb_learning_rate"],
            },
        }
