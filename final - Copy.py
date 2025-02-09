from flask import Flask, render_template, request,  jsonify
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv('C:/Users/dell/OneDrive/Desktop/project/Dataset_Mother.csv')

# Handle missing values
data = dataset.dropna()

# Split the dataset into input features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Scale the features using standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create a Random Forest classifier with 1000 trees
rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_classifier.fit(X, y)

def predict_emergency_situation(mother_age, num_prev_pregnancies, num_miscarriage_stillbirth, diabetes, hypertension, cesarean_section, normal_delivery, gestational_age):
    input_data = np.array([[mother_age, num_prev_pregnancies, num_miscarriage_stillbirth, diabetes, hypertension, cesarean_section, normal_delivery, gestational_age]])
    prediction = rf_classifier.predict(input_data)[0]
    
    if prediction == 1:
        return 'Emergency situation'
    return 'No Emergency situation'

@app.route('/')
def index():
    return render_template('Main_page.html')

@app.route('/predict', methods=['POST'])
def predict():
    mother_age = float(request.form['mother_age'])
    num_prev_pregnancies = int(request.form['num_prev_pregnancies'])
    num_miscarriage_stillbirth = int(request.form['num_miscarriage_stillbirth'])
    diabetes = int(request.form['diabetes'])
    hypertension = int(request.form['hypertension'])
    cesarean_section = int(request.form['cesarean_section'])
    normal_delivery = int(request.form['normal_delivery'])
    gestational_age = int(request.form['gestational_age'])
    
    # Prepare the data to be sent to the server
    data = {
        'mother_age': mother_age,
        'num_prev_pregnancies': num_prev_pregnancies,
        'num_miscarriage_stillbirth': num_miscarriage_stillbirth,
        'diabetes': diabetes,
        'hypertension': hypertension,
        'cesarean_section': cesarean_section,
        'normal_delivery': normal_delivery,
        'gestational_age': gestational_age
    }
    
    # Send a POST request to the server
    response = requests.post('http://example.com/predict', json=data)
    
    # Get the prediction result from the server's response
    result = response.json()['result']
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run()