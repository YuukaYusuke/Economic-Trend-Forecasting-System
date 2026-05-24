# Economic Trend Forecasting System

A practical, understandable time-series forecasting system for currency exchange rates using **BiLSTM + Ensemble** methods with proper time-series validation.

## 🎯 Architecture Overview

### Core Components

```
Data Pipeline (13 features)
        ↓
BiLSTM Model (64→32 units, bidirectional)
        ↓  ↙  ↖
   RF  XGBoost
        ↓  ↙  ↖
   Ensemble (weighted average)
        ↓
Confidence Score (volatility-based)
```

### Key Design Principles

1. **Simplicity**: Understandable, no over-engineering
2. **Time-Series Aware**: TimeSeriesSplit prevents data leakage
3. **Ensemble Approach**: Combines strengths of 3 complementary models
4. **Practical Confidence**: Based on volatility, not complex statistical models
5. **No Production Overhead**: No GARCH, SHAP, LIME, or CI/CD complexity

## 📊 Models

### 1. BiLSTM (Bidirectional LSTM)
- **Role**: Captures sequential patterns in both temporal directions
- **Architecture**: 64 units → 32 units (bidirectional)
- **Features**: 13 engineered features (returns, MAs, volatility, RSI, lags, market stress)
- **Optional**: Attention mechanism for interpretability
- **Weight in Ensemble**: 40%

### 2. Random Forest
- **Role**: Non-linear feature interactions
- **Config**: 200 trees, max_depth=8, min_samples_leaf=5
- **Input**: Last timestep features only
- **Weight in Ensemble**: 30%

### 3. XGBoost
- **Role**: Gradient boosting for residual corrections
- **Config**: 100 trees, max_depth=6, learning_rate=0.1
- **Input**: Last timestep features only
- **Weight in Ensemble**: 30%

## 🔧 Feature Engineering

### Core Features (12)
- `return_1`: 1-day return
- `ma5`, `ma20`, `ma50`: Moving averages
- `volatility_10`, `volatility_20`: Rolling volatility
- `momentum_10`: Price momentum
- `rsi`: Relative Strength Index (14-period)
- `lag_1`, `lag_2`, `lag_5`, `lag_10`: Price lags

### Macroeconomic Feature (1)
- `market_stress_index`: VIX-like indicator based on:
  - Volatility clustering (squared returns)
  - Tail risk (return skewness)
  - Normalized to [0, 1] range
  - **Rationale**: Captures global market stress conditions affecting currency pairs

## 🎓 Training Process

### 1. Data Preparation
```python
raw_data → preprocess → engineer_features → scale & normalize
```

### 2. Time-Series Split (5 folds)
- Prevents look-ahead bias
- Each fold: train on past, validate on future
- No mixing of train/val/test data

### 3. Model Training
```python
# BiLSTM: trained with validation_data
model.fit(X_train, y_train, validation_data=(X_test, y_test))

# RF & XGBoost: trained on each TimeSeriesSplit fold
for train_idx, test_idx in tscv.split(data):
    rf.fit(X[train_idx], y[train_idx])
    xgb.fit(X[train_idx], y[train_idx])
```

### 4. Ensemble Weight Learning
- MSE computed for each model on validation sets
- Weights inversely proportional to MSE
- Normalized to sum to 1
- Example: If BiLSTM MSE=0.0001, RF MSE=0.0002, XGBoost MSE=0.00015
  - BiLSTM weight: 40%, RF weight: 30%, XGBoost weight: 30%

### 5. Confidence Interval Estimation
- Residuals computed: actual - predicted
- 10th and 90th percentiles used for prediction interval
- Stored for each currency model

## 📈 Prediction & Confidence

### Ensemble Prediction
```
pred = 0.40 * bilstm + 0.30 * rf + 0.30 * xgboost
```

### Volatility-Based Confidence
```
confidence_score = 0.60 * vol_component + 0.40 * stress_component

where:
  vol_component = 1 / (1 + recent_volatility)
  stress_component = 1 - market_stress_index

Levels:
  - High: score > 0.7
  - Medium: 0.4 < score < 0.7
  - Low: score < 0.4
```

### Prediction Interval (80%)
```
lower = prediction + residual_q10
upper = prediction + residual_q90
```

## 🚀 Usage

### Training
```bash
# Train all currency models
python train_models.py

# Force retrain (overwrite existing models)
python train_models.py --force

# Custom epochs
python train_models.py --epochs 50
```

### Streamlit App
```bash
streamlit run app.py
```

### Python API
```python
from modules.lstm_model import build_model
from modules.trainer import build_training_bundle
from modules.predict import predict_with_confidence

# Train a currency
bilstm, ensemble, artifacts, data, df = build_training_bundle("euro_to_usd")

# Make predictions
forecast = predict_with_confidence(
    bilstm,
    artifacts,
    live_rate=1.05,
    history=price_history,
    recent_volatility=0.015,
    market_stress=0.35
)

print(forecast["confidence"]["confidence_level"])  # 'high', 'medium', or 'low'
print(forecast["confidence"]["confidence_score"])   # float 0-1
print(forecast["confidence"]["prediction_interval"]) # (lower, upper)
```

## 📁 Project Structure

```
modules/
├── lstm_model.py           # BiLSTM with optional attention
├── ensemble_models.py      # RF + XGBoost + ensemble, TimeSeriesSplit
├── optuna_tuner.py         # Lightweight hyperparameter tuning
├── data_pipeline.py        # Feature engineering + market_stress_index
├── confidence.py           # Volatility-based confidence
├── predict.py              # Ensemble prediction functions
├── trainer.py              # Training pipeline
├── model_storage.py        # Load/save models
└── ... (other modules)
```

## ⚙️ Hyperparameters

### BiLSTM
- Units: [64, 32]
- Dropout: 0.2
- Loss: Huber (robust to outliers)
- Optimizer: Adam

### Random Forest
- n_estimators: 200
- max_depth: 8
- min_samples_leaf: 5

### XGBoost
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8

### Optuna Tuning (Optional)
- Trials: 10-20
- Ranges:
  - RF n_estimators: 50-300
  - RF max_depth: 4-12
  - XGBoost learning_rate: 0.01-0.3

## 🔍 Validation Strategy

### Why TimeSeriesSplit?
- **Standard Cross-Validation**: Leaks future information into training
- **TimeSeriesSplit**: Respects temporal ordering
  - Fold 1: train=[2000-2018], test=[2019]
  - Fold 2: train=[2000-2019], test=[2020]
  - Fold 3: train=[2000-2020], test=[2021]
  - ...

### Model Performance Metrics
- MSE (Mean Squared Error): Computed per fold
- Prediction Interval Coverage: % of actuals within [lower, upper]
- Confidence Calibration: % high-confidence predictions with low errors

## 🎨 Confidence Visualization

The Streamlit app shows:
1. **Confidence Badge**: High/Medium/Low indicator
2. **Confidence Score**: Numerical value (0-1)
3. **Prediction Interval**: Upper and lower bounds
4. **Volatility Alert**: Flag if market volatility is high
5. **Market Stress**: Current stress index

## 📚 References

### Data Pipeline
- `engineer_features()`: Creates 12 technical + 1 macro feature
- `_market_stress_index()`: Volatility clustering + skewness-based stress
- Feature scaling: StandardScaler on training set only (no leakage)

### TimeSeriesSplit
- `sklearn.model_selection.TimeSeriesSplit`
- Ensures no look-ahead bias
- 5 folds by default (configurable)

### Ensemble Weights
- Learned from validation MSE
- Inverse relationship: lower MSE → higher weight
- Normalized to sum to 1

## 🔗 Dependencies

Core:
- TensorFlow/Keras (BiLSTM)
- scikit-learn (RF, preprocessing, TimeSeriesSplit)
- pandas, numpy (data handling)
- XGBoost (gradient boosting)

Optional:
- Optuna (hyperparameter tuning)
- Streamlit (visualization)

## 💡 Design Decisions

### Why 3 Models?
1. BiLSTM: Temporal patterns
2. RF: Non-linear interactions
3. XGBoost: Residual corrections
→ Diversification reduces overfitting

### Why No Transformer?
- Simpler is better for interpretability
- BiLSTM sufficient for daily forecasting
- Lower computational cost

### Why No GARCH?
- Volatility-based confidence simpler and practical
- Residual quantiles sufficient for prediction intervals
- Avoids over-specification

### Why No SHAP/LIME?
- Adds complexity without clear business value
- Ensemble weights provide interpretability
- Feature importance in data pipeline is clear

### Why Huber Loss?
- Robust to outliers
- More stable than MSE for financial data
- Smooth gradient (better for optimization)

## 🎯 Next Steps (Optional Enhancements)

1. **Adaptive Weights**: Update ensemble weights based on recent performance
2. **Feature Importance**: Use permutation importance from ensemble
3. **Online Learning**: Retrain models incrementally as new data arrives
4. **Multi-Step Ahead**: Predict 5-day or 10-day horizons
5. **Ensemble Stacking**: Use meta-learner for weight optimization

## ⚠️ Constraints & Limitations

- **Single Macroeconomic Feature**: Market stress index only (by design)
- **No External Data**: Uses only historical prices
- **Daily Forecasts**: Optimized for 1-day ahead
- **In-Sample Only**: No real-time data provider integration
- **Single Currency Pair Format**: Exchange rates to USD

## 📝 License

This implementation follows the project's existing license.

---

**Version**: 2.0 (BiLSTM + Ensemble with TimeSeriesSplit)
**Last Updated**: 2026-05-25
