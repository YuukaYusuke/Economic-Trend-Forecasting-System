from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential


def build_model(input_shape):
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss=Huber())
    return model
