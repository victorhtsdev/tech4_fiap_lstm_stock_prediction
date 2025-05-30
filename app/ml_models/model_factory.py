from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class ModelFactory(ABC):
    @abstractmethod
    def create_model(self, input_shape):
        pass

class LSTMModelFactory(ModelFactory):
    def create_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True),
            Dropout(0.1),
            LSTM(128, return_sequences=True),
            Dropout(0.1),
            LSTM(128),
            Dropout(0.1),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model