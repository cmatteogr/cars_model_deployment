from typing import Callable
from fastapi import FastAPI
from services.model import MLModel
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.ensemble import IsolationForest
from gensim.models import Word2Vec
import os
import json
import joblib


def _startup_model(app: FastAPI, model_path: str) -> None:
    # Read model data file
    model_data_filepath = os.path.join(model_path, 'cars_regressor_price_model_data.json')
    with open(model_data_filepath, 'r') as json_file:
        model_data = json.load(json_file)
    # Load preprocess models
    preprocess_data = model_data['preprocess_config_data']
    pp_models = preprocess_data['models_filenames']
    # Load Hasher Encoder model
    hasher_model: FeatureHasher = joblib.load(os.path.join(model_path, pp_models['hasher_model_model_filename']))
    # Load drivetrain Encoder model
    ohe_drivetrain: OneHotEncoder = joblib.load(os.path.join(model_path, pp_models['ohe_drivetrain_model_filename']))
    # Load make Encoder model
    ohe_make: OneHotEncoder = joblib.load(os.path.join(model_path, pp_models['ohe_make_model_filename']))
    # Load Body style Encoder model
    ohe_bodystyle: OneHotEncoder = joblib.load(os.path.join(model_path, pp_models['ohe_bodystyle_model_filename']))
    # Load Fuel type Encoder model
    ohe_fuel_type: OneHotEncoder = joblib.load(os.path.join(model_path, pp_models['ohe_fuel_type_model_filename']))
    # Load Exterior color Encoder model
    w2v_exterior_color = Word2Vec.load(os.path.join(model_path, pp_models['w2v_exterior_color_model_filename']))
    # Load Interior color Encoder model
    w2v_interior_color = Word2Vec.load(os.path.join(model_path, pp_models['w2v_interior_color_model_filename']))
    # Load Cat Encoder model
    w2v_cat = Word2Vec.load(os.path.join(model_path, pp_models['w2v_cat_model_filename']))
    # Load imputer model
    imputer: IterativeImputer = joblib.load(os.path.join(model_path, pp_models['imputer_model_filename']))
    # Load outlier detector model
    outlier_detector: IsolationForest = joblib.load(os.path.join(model_path, pp_models['outlier_detection_filename']))
    # Load scaler model
    scaler: MinMaxScaler = joblib.load(os.path.join(model_path, pp_models['scaler_model_filename']))

    # Init the ML object
    # NOTE: For this demo we are using all the price prediction models, pick one in you case, update the following code
    # Init random forest model
    model_randomforest_instance = MLModel('random_forest_model_cars_price_prediction.pkl', 'randomforest')
    model_catboost_instance = MLModel('cat_boost_model_cars_price_prediction.pkl', 'catboost')
    model_automl_instance = MLModel('automl_model_cars_price_prediction', 'automl')
    model_neural_network_tensorflow_instance = MLModel('neural_network_tensorflow_model_cars_price_prediction.keras',
                                                       'neural_network_tensorflow')
    model_neural_network_pytorch_instance = MLModel('neural_network_pytorch_model_cars_price_prediction.pth',
                                                    'neural_network_pytorch')

    model_dict = {
        'preprocess': {
            'price_threshold': preprocess_data['preprocess_config']['price_threshold'],
            'make_valid_categories': preprocess_data['preprocess_config']['make_valid_categories'],
            'exterior_color_vector_size': preprocess_data['preprocess_config']['exterior_color_vector_size'],
            'interior_color_vector_size': preprocess_data['preprocess_config']['interior_color_vector_size'],
            'cat_vector_size': preprocess_data['preprocess_config']['cat_vector_size'],
            'hasher_model_model_filename': hasher_model,
            'ohe_drivetrain_model': ohe_drivetrain,
            'ohe_make_model': ohe_make,
            'ohe_bodystyle_model': ohe_bodystyle,
            'ohe_fuel_type_model': ohe_fuel_type,
            'w2v_exterior_color_model': w2v_exterior_color,
            'w2v_interior_color_model': w2v_interior_color,
            'w2v_cat_model': w2v_cat,
            'imputer_model': imputer,
            'outlier_detector_model': outlier_detector,
            'scaler_model': scaler,
        },
        'predict': {
            'model_randomforest_instance': model_randomforest_instance,
            'model_catboost_instance': model_catboost_instance,
            'model_automl_instance': model_automl_instance,
            'model_neural_network_tensorflow_instance': model_neural_network_tensorflow_instance,
            'model_neural_network_pytorch_instance': model_neural_network_pytorch_instance,
        }
    }

    # Add to app state model
    app.state.model = model_dict


def _shutdown_model(app: FastAPI) -> None:
    app.state.model = None


def start_app_handler(app: FastAPI, model_path: str) -> Callable:
    def startup() -> None:
        _startup_model(app, model_path)

    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        _shutdown_model(app)

    return shutdown
