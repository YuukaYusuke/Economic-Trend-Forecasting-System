"""
Unit tests for prediction module.
Run with: pytest tests/test_predict.py -v
"""

import pytest
import numpy as np
from modules.predict import (
    ensemble_return,
    _validate_numeric,
    predict_rf_return,
    predict_xgb_return,
)


class TestValidateNumeric:
    """Test input validation function."""
    
    def test_valid_float(self):
        """Test with valid float."""
        assert _validate_numeric(0.5, "test") == True
    
    def test_valid_int(self):
        """Test with valid int."""
        assert _validate_numeric(5, "test") == True
    
    def test_valid_numpy_float(self):
        """Test with numpy float."""
        assert _validate_numeric(np.float32(0.5), "test") == True
    
    def test_nan_raises_error(self):
        """Test NaN raises ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            _validate_numeric(float('nan'), "test")
    
    def test_inf_raises_error(self):
        """Test Inf raises ValueError."""
        with pytest.raises(ValueError, match="Inf"):
            _validate_numeric(float('inf'), "test")
    
    def test_none_allowed(self):
        """Test None is allowed with allow_none=True."""
        assert _validate_numeric(None, "test", allow_none=True) == True
    
    def test_none_not_allowed(self):
        """Test None raises error with allow_none=False."""
        with pytest.raises(TypeError):
            _validate_numeric(None, "test", allow_none=False)
    
    def test_string_raises_error(self):
        """Test string raises TypeError."""
        with pytest.raises(TypeError):
            _validate_numeric("0.5", "test")


class TestEnsembleReturn:
    """Test ensemble_return function."""
    
    def test_all_models_equal_return(self):
        """Test with all models returning same value."""
        result = ensemble_return(0.01, 0.01, 0.01)
        assert result == pytest.approx(0.01)
    
    def test_weighted_average(self):
        """Test weighted average calculation."""
        # LSTM: 0.01 * 0.4, RF: 0.02 * 0.3, XGB: 0.03 * 0.3
        # = 0.004 + 0.006 + 0.009 = 0.019
        result = ensemble_return(0.01, 0.02, 0.03)
        expected = (0.01 * 0.4 + 0.02 * 0.3 + 0.03 * 0.3) / 1.0
        assert result == pytest.approx(expected)
    
    def test_only_lstm(self):
        """Test with only LSTM (RF and XGB are None)."""
        result = ensemble_return(0.01, None, None)
        assert result == pytest.approx(0.01)
    
    def test_lstm_and_rf(self):
        """Test with LSTM and RF only."""
        result = ensemble_return(0.01, 0.015, None)
        # Weight recalculates: LSTM: 0.4, RF: 0.3, total: 0.7
        # = (0.01 * 0.4 + 0.015 * 0.3) / 0.7
        expected = (0.01 * 0.4 + 0.015 * 0.3) / 0.7
        assert result == pytest.approx(expected)
    
    def test_lstm_and_xgb(self):
        """Test with LSTM and XGB only."""
        result = ensemble_return(0.01, None, 0.015)
        # Weight recalculates: LSTM: 0.4, XGB: 0.3, total: 0.7
        expected = (0.01 * 0.4 + 0.015 * 0.3) / 0.7
        assert result == pytest.approx(expected)
    
    def test_custom_weights(self):
        """Test with custom weights."""
        weights = {"bilstm": 0.5, "rf": 0.3, "xgb": 0.2}
        result = ensemble_return(0.01, 0.02, 0.03, weights=weights)
        expected = (0.01 * 0.5 + 0.02 * 0.3 + 0.03 * 0.2) / 1.0
        assert result == pytest.approx(expected)
    
    def test_negative_returns(self):
        """Test with negative returns."""
        result = ensemble_return(-0.01, -0.02, -0.03)
        expected = (-0.01 * 0.4 - 0.02 * 0.3 - 0.03 * 0.3) / 1.0
        assert result == pytest.approx(expected)
    
    def test_mixed_returns(self):
        """Test with mixed positive and negative returns."""
        result = ensemble_return(0.01, -0.01, 0.02)
        expected = (0.01 * 0.4 - 0.01 * 0.3 + 0.02 * 0.3) / 1.0
        assert result == pytest.approx(expected)
    
    def test_nan_input_raises_error(self):
        """Test NaN input raises error."""
        with pytest.raises(ValueError):
            ensemble_return(float('nan'), 0.01, 0.01)
    
    def test_inf_input_raises_error(self):
        """Test Inf input raises error."""
        with pytest.raises(ValueError):
            ensemble_return(float('inf'), 0.01, 0.01)


class TestPredictRFReturn:
    """Test predict_rf_return function."""
    
    def test_none_model_returns_none(self):
        """Test None model returns None."""
        result = predict_rf_return(None, None, None)
        assert result is None
    
    def test_none_scaler_raises_error(self):
        """Test None scaler raises error."""
        # Create mock model
        class MockModel:
            def predict(self, X):
                return np.array([[0.01]])
        
        model = MockModel()
        with pytest.raises(ValueError):
            predict_rf_return(model, None, np.array([[1, 2, 3]]))


class TestPredictXGBReturn:
    """Test predict_xgb_return function."""
    
    def test_none_model_returns_none(self):
        """Test None model returns None."""
        result = predict_xgb_return(None, None, None)
        assert result is None
    
    def test_none_scaler_raises_error(self):
        """Test None scaler raises error."""
        class MockModel:
            def predict(self, X):
                return np.array([[0.01]])
        
        model = MockModel()
        with pytest.raises(ValueError):
            predict_xgb_return(model, None, np.array([[1, 2, 3]]))


class TestEnsembleIntegration:
    """Integration tests for ensemble predictions."""
    
    def test_realistic_scenario(self):
        """Test realistic prediction scenario."""
        # Simulate 3 models making predictions
        lstm_pred = 0.0125
        rf_pred = 0.0118
        xgb_pred = 0.0132
        
        result = ensemble_return(lstm_pred, rf_pred, xgb_pred)
        
        # Result should be weighted average
        assert 0.011 < result < 0.014
        assert isinstance(result, float)
    
    def test_extreme_volatility(self):
        """Test with extreme volatility predictions."""
        lstm_pred = -0.05
        rf_pred = 0.05
        xgb_pred = -0.03
        
        result = ensemble_return(lstm_pred, rf_pred, xgb_pred)
        
        # Should still compute valid weighted average
        assert -0.05 <= result <= 0.05
        assert not np.isnan(result)
        assert not np.isinf(result)


if __name__ == "__main__":
    # Can run with: python -m pytest tests/test_predict.py -v
    pytest.main([__file__, "-v"])
