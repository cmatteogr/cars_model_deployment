from typing import List
from pydantic import BaseModel


class CarData(BaseModel):
    msrp: int
    year: int
    model: str
    interior_color: str
    drivetrain: str
    mileage: int
    make: str
    bodystyle: str
    cat: str
    fuel_type: str
    stock_type: str
    exterior_color: str


class PredictRequest(BaseModel):
    model_type: str
    cars_data: List[CarData]
    cars_real_price: List[int]


class PredictResponse(BaseModel):
    result: List[float]
