from abc import ABC, abstractmethod
from typing import Any
from keras.models import load_model
import os
from pycaret.regression import load_model
from catboost import CatBoostRegressor
import pickle
import torch
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor


class BaseMLModel(ABC):
    @abstractmethod
    def predict(self, req: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _load_model(self, req: Any) -> Any:
        raise NotImplementedError


class MLModel(BaseMLModel):
    """Sample ML model"""

    def __init__(self, model_path: str, model_tool: str) -> None:
        self.model_tool = model_tool
        self.model = self._load_model(model_path)
        self.model_path = model_path

    def predict(self, cars_df) -> [float]:

        # Apply prediction
        match self.model_tool:
            case 'randomforest':
                y_pred = self.model.predict(cars_df.values)
            case 'catboost':
                y_pred = self.model.predict(cars_df.values)
            case 'automl':
                y_pred = self.model.predict(cars_df)
            case 'neural_network_tensorflow':
                y_pred = self.model.predict(cars_df, verbose=0)
            case 'neural_network_pytorch':
                self.model.eval()
                # Transform input to tensor
                X_tensor = torch.tensor(cars_df.values, dtype=torch.float32)
                # Predict test dataset
                y_pred = self.model(X_tensor)
                y_pred = y_pred.detach().numpy()
            case _:
                raise Exception(f"Invalid Testing model: {self.model_tool}")
        return y_pred

    def _load_model(self, model_path: str) -> Any:
        # Test
        model_path = os.path.join('ml_model', model_path)
        match self.model_tool:
            case 'randomforest':
                with open(model_path, 'rb') as file:
                    regression_cars_price_model: RandomForestRegressor = pickle.load(file)
            case 'catboost':
                with open(model_path, 'rb') as file:
                    regression_cars_price_model: CatBoostRegressor = pickle.load(file)
            case 'automl':
                regression_cars_price_model = load_model(model_path)
            case 'neural_network_tensorflow':
                regression_cars_price_model = tf.keras.models.load_model(model_path)
            case 'neural_network_pytorch':
                regression_cars_price_model = torch.load(model_path)
            case _:
                raise Exception(f"Invalid Testing model: {model_path}")

        return regression_cars_price_model
