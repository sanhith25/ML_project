from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
    
        gender = request.form.get('gender')
        ethnicity = request.form.get('ethnicity')
        parental_education = request.form.get('parental_level_of_education')
        lunch = request.form.get('lunch')
        test_prep = request.form.get('test_preparation_course')
        writing_score = float(request.form.get('writing_score'))
        reading_score = float(request.form.get('reading_score'))

    
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_education,
            lunch=lunch,
            test_preparation_course=test_prep,
            reading_score=reading_score,
            writing_score=writing_score
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        inputs = {
            'Gender': gender,
            'Ethnicity': ethnicity,
            'Parental Education': parental_education,
            'Lunch': lunch,
            'Test Preparation': test_prep,
            'Writing Score': writing_score,
            'Reading Score': reading_score
        }

        return render_template('home.html', results=results[0], inputs=inputs)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
