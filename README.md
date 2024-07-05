# Cars Price Prediction - Model Deployment

![Machine_Learning_Inference_Header_f0ec9b25f1](https://github.com/cmatteogr/cars_model_deployment/assets/138587358/5093bf9e-e724-44da-bfb6-79538d00476e)

Once the model is trained with good performance metrics as a result, then it's time to deploy it in a production environment. This repo shows a simple way to do it.

Our use case is the [Cars Price Prediction - US Market](https://github.com/cmatteogr/cars_ml_project), the models trained from this stages are used for the inference (Random Forest, CatBoost, AutoML, Neural Networs), that way we can check the different inference challenges: Infrastructure, Cost, Latency, Size, etc.

## Prerequisites
* Install Python 3.11
* Install the libraries using requirements.txt.
* Add the trained models generated from [Cars Price Prediction - US Market](https://github.com/cmatteogr/cars_ml_project) in the ml_model folder, include the preprocess models and model_data.json file
* To use the service import the postman_collection/cars_ml_inference.postman_collection.json file in Postman, it contains the structure needed to use the service

## Usage
Execute the script main.py to start the service, it will read the preprocess and prediction models/configurations to start the model inference. Once the model is started, use the Postman service to execute the model inference (service) and get the predicted cars prices.
