import sklearn
sklearn.set_config(transform_output='pandas')

import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
import logging
import src.logger
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_focus_index=True):
        self.add_focus_index = add_focus_index
        self.feature_names_in_ = []

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        X_df = X.copy()

        if 'id' in X_df.columns:
            logging.info("Dropping id column...")
            X_df.drop(columns=['id'], inplace=True)

        if self.add_focus_index:
            logging.info("add_focus_index == True, adding new features...")
            X_df['academic_effort'] = X_df['study_hours'] * X_df['class_attendance']
            X_df['academic_focus_seconds'] = (X_df['study_hours'] / X_df['class_attendance']) * 3600
            X_df['productivity'] = X_df['study_hours'] * X_df['sleep_hours']

        return X_df
    
    def get_feature_names_out(self, input_features=None):
        logging.info("Getting feature names...")

        if input_features is None or len(input_features) == 0:
            feature_names = self.feature_names_in_.copy()
        else:
            feature_names = list(input_features)

        if 'id' in feature_names:
            logging.info("Dropping id feature name...")
            feature_names.remove('id')

        if self.add_focus_index:
            logging.info("add_focus_index == True, adding new feature names...")
            feature_names.extend(['academic_effort', 'academic_focus_seconds', 'productivity'])

        return np.array(feature_names)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("feature_adder", CustomFeatureAdder()),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Num Pipeline Initiated, Feature Adder Initiated")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Cat Pipeline Initiated, One Hot Encoder Initiated")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, make_column_selector(dtype_include=np.number)),
                    ("cat_pipeline", cat_pipeline, make_column_selector(dtype_include=object)),
                ]
            )

            logging.info("Preprocessor Pipeline Initiated, Column Transformer with automatic feature selection initiated")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            preprocessor = self.get_data_transformer_object()

            logging.info("Data Transformation Initiated")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            input_feature_train_df = train_df.drop(columns=['exam_score'], axis=1)
            target_feature_train_df = train_df['exam_score']

            input_feature_test_df = test_df.drop(columns=['exam_score'], axis=1)
            target_feature_test_df = test_df['exam_score']

            logging.info("Train Test Split Initiated")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            feature_names = preprocessor.get_feature_names_out()
            logging.info(f"Data Transformation Completed with feature names: {feature_names}")

            train_arr = np.c_[np.array(input_feature_train_arr), np.array(target_feature_train_df)]
            test_arr = np.c_[np.array(input_feature_test_arr), np.array(target_feature_test_df)]

            logging.info("Preprocessing Pipeline Completed")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f"Error occured at Data Transformation stage: {e}")
            raise CustomException(e, sys)