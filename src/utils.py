import os
import sys

import numpy as np
import pandas as pd

import dill

from src.exception import CustomException
import logging
import src.logger

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