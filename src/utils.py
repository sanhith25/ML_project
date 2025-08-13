import sys
import os
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, Y_train, X_test, Y_test, models): ##hyperparameter tuning for all the models 
   
    try:
        report = {}

        param_grids = {
            "catboost": {
                "depth": [6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "iterations": [200, 500]
            },
            "xgboost": {
                "n_estimators": [100, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7]
            },
            "randomforest": {
                "n_estimators": [100, 200, 500],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
            "gradientboosting": {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7]
            }
        }

        for name, model in models.items():
            logging.info(f"Training model: {name}")
            model_key = name.lower()

            if model_key in param_grids:
                logging.info(f"Tuning hyperparameters for {name}...")
                gs = GridSearchCV(model, param_grids[model_key], cv=3, scoring="r2", n_jobs=-1, verbose=1)
                gs.fit(X_train, Y_train)
                model = gs.best_estimator_
                logging.info(f"Best params for {name}: {gs.best_params_}")
            else:
                logging.info(f"No tuning grid found for {name}, fitting directly.")
                model.fit(X_train, Y_train)

            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)

            train_model_score = r2_score(Y_train, Y_train_pred)
            test_model_score = r2_score(Y_test, Y_test_pred)

            logging.info(f"{name} Train R²: {train_model_score}, Test R²: {test_model_score}")

            report[name] = {
                "train_r2": train_model_score,
                "test_r2": test_model_score,
                "model": model
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
