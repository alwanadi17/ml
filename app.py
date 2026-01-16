from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            id = 0
            age = int(request.form.get('age'))
            gender = request.form.get('gender')
            course = request.form.get('course')
            study_hours = float(request.form.get('study_hours'))
            class_attendance = float(request.form.get('class_attendance'))
            internet_access = request.form.get('internet_access')
            sleep_hours = float(request.form.get('sleep_hours'))
            sleep_quality = request.form.get('sleep_quality')
            study_method = request.form.get('study_method')
            facility_rating = request.form.get('facility_rating')
            exam_difficulty = request.form.get('exam_difficulty')

            data = CustomData(id, age, gender, course, study_hours, class_attendance, internet_access, sleep_hours, sleep_quality, study_method, facility_rating, exam_difficulty)
            data_df = data.get_data_as_dataframe()

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(data_df)

            return render_template('index.html', results=results[0])
        
        except Exception as e:
            return f"An error occurred: {e}"

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)