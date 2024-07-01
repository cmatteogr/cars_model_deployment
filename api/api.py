from typing import Any
from fastapi import APIRouter, Request
from models.predict import PredictRequest, PredictResponse
from services.preprocess import preprocess
import time
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
    # Get model type and cars data
    model_type = payload.model_type
    print(f"Predict cars prices using: {model_type}")
    cars_data_json = [car.dict() for car in payload.cars_data]
    cars_df = pd.DataFrame.from_dict(cars_data_json)
    print(f"Number of instances: {cars_df.shape[0]}")

    # Define if scale data is needed
    scale_data = model_type in ['neural_network_tensorflow', 'neural_network_pytorch']

    # Apply preprocess
    preprocess_start_time = time.time()
    cars_df = preprocess(cars_df,
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
    print(f"Preprocess time: {(time.time() - preprocess_start_time)}")

    # Apply prediction
    predicts_start_time = time.time()
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

    # Predict
    predict_value = model.predict(cars_df)
    print(f"Predict time: {(time.time() - predicts_start_time)}")

    # Return prediction
    return PredictResponse(result=predict_value)
