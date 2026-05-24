"""
BiLSTM model with optional attention mechanism for time-series forecasting.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Input,
    Attention,
    Concatenate,
    Flatten,
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Model, Sequential


def build_model(input_shape, use_attention=False):
    """
    Build a BiLSTM model for time-series forecasting.
    
    Args:
        input_shape: (window_size, num_features)
        use_attention: If True, add simple attention mechanism
    
    Returns:
        Compiled Keras model
    """
    if not use_attention:
        # Standard BiLSTM without attention
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss=Huber())
        return model
    
    # BiLSTM with attention
    inputs = Input(shape=input_shape)
    
    # BiLSTM layer
    bilstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    bilstm_out = Dropout(0.2)(bilstm_out)
    
    # Simple attention: multiply sequence by learned weights
    # Context vector from second BiLSTM layer
    context = Bidirectional(LSTM(32, return_sequences=True))(bilstm_out)
    context = Dropout(0.2)(context)
    
    # Compute attention scores (simplified: use last context as query)
    query = LSTM(32, return_sequences=False)(context)
    query = tf.expand_dims(query, axis=1)  # (batch, 1, 32)
    
    # Attention between query and context
    attn = Attention()([query, context])  # (batch, 1, 64)
    attn = Flatten()(attn)  # (batch, 64)
    
    # Combine with BiLSTM output
    final = Bidirectional(LSTM(32, return_sequences=False))(bilstm_out)
    final = Concatenate()([final, attn])
    final = Dropout(0.2)(final)
    final = Dense(16, activation="relu")(final)
    outputs = Dense(1)(final)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=Huber())
    return model
