from typing import Any
from fastapi import APIRouter, Request
from models.predict import PredictRequest, PredictResponse
from services.preprocess import preprocess
import json
import pandas as pd

api_router = APIRouter()


@api_router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, payload: PredictRequest) -> Any:
    """
    ML Prediction API
    """
    # Get model data
    model_data = request.app.state.model
    model_preprocess_data = model_data['preprocess']

    request_data = json.loads(payload.input_text)
    model_type = request_data['model_type']
    cars_data = request_data['cars_data']

    cars_df = pd.read_json(cars_data)

    #

    scale_data = model_type in ['neural_network_tensorflow', 'neural_network_pytorch']

    # Apply preprocess
    preprocess(cars_df,
               model_preprocess_data['make_valid_categories'],
               model_preprocess_data['hasher_model_model_filename'],
               model_preprocess_data['exterior_color_vector_size'],
               model_preprocess_data['w2v_exterior_color_model'],
               model_preprocess_data['interior_color_vector_size'],
               model_preprocess_data['w2v_interior_color_model'],
               model_preprocess_data['ohe_make_model'],
               model_preprocess_data['ohe_drivetrain_model'],
               model_preprocess_data['ohe_bodystyle_model'],
               model_preprocess_data['cat_vector_size'],
               model_preprocess_data['w2v_cat_model'],
               model_preprocess_data['ohe_fuel_type_model'],
               model_preprocess_data['imputer_model'],
               model_preprocess_data['outlier_detector_model'],
               model_preprocess_data['scaler_model'],
               scale_data=scale_data)

    model_predict_data = model_data['predict']
    match model_type:
        case 'randomforest':
            model = model_predict_data['model_randomforest_instance']
        case 'catboost':
            model = model_predict_data['model_catboost_instance']
        case 'automl':
            model = model_predict_data['model_automl_instance']
        case 'neural_network_tensorflow':
            model = model_predict_data['model_neural_network_tensorflow_instance']
        case 'neural_network_pytorch':
            model = model_predict_data['model_neural_network_pytorch_instance']
        case _:
            raise Exception("Model type not supported")

    predict_value = model.predict(cars_df)
    return PredictResponse(result=predict_value)
