from src.logger import logging
from src.exception import CustomException
import traceback
import os
import sys

from src.utils import load_object, save_user_input
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, data):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            features = pd.DataFrame(data)
            cols_order = ['id', 'age', 'gender', 'course', 'study_hours', 
                      'class_attendance', 'internet_access', 'sleep_hours', 
                      'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']
            features = features[cols_order]

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)

            tmp = features.copy()

            tmp['exam_score'] = preds
            features_dict = tmp.to_dict(orient='records')[0]
            save_user_input(features_dict)
            return preds

        except Exception as e:
            logging.error(traceback.format_exc())
            logging.error(f"Error occured in prediction pipeline: {e}")
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                 id: int,
                 age: int,
                 gender: object,
                 course: object,
                 study_hours: float,
                 class_attendance: float,
                 internet_access: object,
                 sleep_hours: float,
                 sleep_quality: object,
                 study_method: object,
                 facility_rating: object,
                 exam_difficulty: object):
        self.id = id
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
                "id": [self.id],
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