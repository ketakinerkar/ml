import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            print(f"Tuning hyperparameters for {model_name}...")

            # Get parameters for the model
            para = param.get(model_name, {})

            if para:  # If there are parameters to tune
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)

                # Use the best model with tuned parameters
                best_model = gs.best_estimator_
            else:
                best_model = model  # If no parameters to tune, use default model

            # Train the final model
            best_model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Compute R² scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test model score
            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys) 