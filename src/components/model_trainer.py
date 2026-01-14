import logging
import src.logger
from src.exception import CustomException
import os
import sys

from dataclasses import dataclass
from src.utils import save_object, evaluate_models

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
import yaml
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Hist Gradient Boosting": HistGradientBoostingRegressor(),
                "Linear Regression": LinearRegression(n_jobs=-1),
                "XGBRegressor": XGBRegressor(n_jobs=-1, device='cuda', tree_method='hist'),
                "CatBoost Regressor": CatBoostRegressor(verbose=False, thread_count=-1, task_type="GPU", devices='0'),
            }

            with open("config/params.yaml", "r") as file:
                all_params = yaml.safe_load(file)

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                models=models, params=all_params)

            best_model_name = max(model_report, key=lambda x: model_report[x]['best_score'])
            best_model = model_report[best_model_name]['best_model']
            best_model_score = model_report[best_model_name]['best_score']


            if best_model_score < 0.6:
                raise CustomException("No best model found with r2_score greater than the threshold 0.6", sys)

            logging.info(f"Best model found: {best_model_name} with r2_score: {best_model_score}")
            logging.info(f"is tuned better: {model_report[best_model_name]['is_tuned_better']}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_model_score

        except Exception as e:
            logging.error(f"Error occured at model trainer stage: {e}")
            raise CustomException(e, sys)
        