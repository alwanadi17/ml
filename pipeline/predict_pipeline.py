from src.logger import logging
from src.exception import CustomException
import os
import sys

from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            logging.error(f"Error occured in prediction pipeline: {e}")
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                 age: float,
                 gender: float,
                 course: float,
                 study_hours: float,
                 class_attendance: float,
                 internet_access: float,
                 sleep_hours: float,
                 sleep_quality: float,
                 study_method: float,
                 facility_rating: float,
                 exam_difficulty: float):
        self.age = age
        self.gender = gender
        self.course = course
        self.study_hours = study_hours
        self.class_attendance = class_attendance
        self.internet_access = internet_access
        self.sleep_hours = sleep_hours
        self.sleep_quality = sleep_quality
        self.study_method = study_method
        self.facility_rating = facility_rating
        self.exam_difficulty = exam_difficulty

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "gender": [self.gender],
                "course": [self.course],
                "study_hours": [self.study_hours],
                "class_attendance": [self.class_attendance],
                "internet_access": [self.internet_access],
                "sleep_hours": [self.sleep_hours],
                "sleep_quality": [self.sleep_quality],
                "study_method": [self.study_method],
                "facility_rating": [self.facility_rating],
                "exam_difficulty": [self.exam_difficulty],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Custom data frame created successfully")
            return df

        except Exception as e:
            logging.error(f"Error occured in get_data_as_dataframe: {e}")
            raise CustomException(e, sys)