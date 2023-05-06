import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_Data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))


@app.route('/')
def index():
    location = sorted(data['location'].unique())
    bhk = sorted(data['bhk'].unique())
    bath = sorted(data['bath'].unique())
    total = sorted(data['total_sqft'].unique())
    return render_template('index.html', location=location, bhk=bhk, bath=bath, total_sqft=total)


@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    total = request.form.get('total_sqft')

    print(location, bhk, bath, total)
    input = pd.DataFrame([[location, total, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input)[0] * 100000

    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5006)
