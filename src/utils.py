import os
import sys

import numpy as np
import pandas as pd

import dill

from src.exception import CustomException
import logging
import src.logger

from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        logging.error(f"Error occured at save_object stage: {e}")
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        logging.info("Starting model evaluation")
        export = []
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]

            param = params[model_name]

            logging.info(f"{model_name} vanilla fitting...")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            vanilla_score = r2_score(y_test, y_pred)

            logging.info(f"{model_name} vanilla r2 score: {vanilla_score}")
            logging.info(f"{model_name} hyperparameter tuning fitting...")

            rs = RandomizedSearchCV(model, param, cv=5, n_iter=10, n_jobs=4, verbose=0)
            rs.fit(X_train, y_train)

            best_model = rs.best_estimator_

            y_pred_tuned = best_model.predict(X_test)
            tuned_score = r2_score(y_test, y_pred_tuned)

            logging.info(f"{model_name} tuned r2 score: {tuned_score}")

            result = {
                'model': model_name,
                'vanilla_test_score': vanilla_score,
                'best_parameters': rs.best_params_,
                'best_rs_score': rs.best_score_,
                'tuned_test_score': tuned_score
            }
            export.append(result)

            report[model_name] = {
                'vanilla_score': vanilla_score,
                'tuned_score': tuned_score,
                'best_score': max(vanilla_score, tuned_score),
                'is_tuned_better': tuned_score > vanilla_score,
                'best_model': best_model,
            }

        logging.info("Model evaluation completed")
        logging.info(f"Exporting to csv...")

        export = pd.DataFrame(export)
        export = export.sort_values(by='tuned_test_score', ascending=False)
        export_path = os.path.join("artifacts", "model_report.csv")
        export.to_csv(export_path, index=False)

        logging.info(f"Model report saved at {export_path}")

        return report
    
    except Exception as e:
        logging.error(f"Error occured at evaluate_models stage: {e}")
        raise CustomException(e, sys)