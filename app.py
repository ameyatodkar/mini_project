import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('gradient_boosting.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 
                'InterestRate', 'DTIRatio', 'Education', 'EmploymentType', 
                'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

    int_features = [int(request.form[feature]) for feature in features]
    df = pd.DataFrame([int_features], columns=features)
    
    # Assuming your model predicts 1 for default and 0 for non-default
    prediction = model.predict(df)

    output = "Loan Default" if prediction == 1 else "Loan Not Default"

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
     #  app.run(debug=True)

