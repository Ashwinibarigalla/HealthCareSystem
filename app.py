from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load Machine Learning Models
dia_model = pickle.load(open('Models/diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('Models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('Models/parkinsons_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        # Capture user inputs for diabetes prediction
        Pregnancies = float(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['SkinThickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = float(request.form['Age'])

        # Make predictions using the diabetes model
        features = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        prediction = dia_model.predict([features])[0]

        if prediction == 0:
            result = "The person is not likely to have diabetes."
        else:
            result = "The person is likely to have diabetes."

        return render_template('diabetes.html', result=result)

    return render_template('diabetes.html', result=None)


@app.route('/heart_disease', methods=['GET', 'POST'])
def heart_disease():
    if request.method == 'POST':
        # Getting the Features from the User
        age = int(request.form['age'])
        sex = request.form['sex']
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Preprocessing the Features
        if sex == 'Male':
            sex = 1
        else:
            sex = 0

        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        prediction = heart_model.predict([features])[0]

        if prediction == 0:
            result = "The person is not likely to have heart disease."
        else:
            result = "The person is likely to have heart disease."

        return render_template('heart_disease.html', result=result)

    return render_template('heart_disease.html', result=None)

@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    if request.method == 'POST':
        # Capture user inputs for Parkinson's disease prediction
        MDVP_Fo = float(request.form['MDVP_Fo'])
        MDVP_Fhi = float(request.form['MDVP_Fhi'])
        MDVP_Flo = float(request.form['MDVP_Flo'])
        MDVP_Jitter = float(request.form['MDVP_Jitter'])
        MDVP_Jitter_Abs = float(request.form['MDVP_Jitter_Abs'])
        RAP = float(request.form['RAP'])
        PPQ = float(request.form['PPQ'])
        DDP = float(request.form['DDP'])
        MDVP_Shimmer = float(request.form['MDVP_Shimmer'])
        shimmer_dB = float(request.form['shimmer_dB'])
        APQ3 = float(request.form['APQ3'])
        APQ5 = float(request.form['APQ5'])
        APQ = float(request.form['APQ'])
        DDA = float(request.form['DDA'])
        NHR = float(request.form['NHR'])
        HNR = float(request.form['HNR'])
        RPDE = float(request.form['RPDE'])
        DFA = float(request.form['DFA'])
        spread1 = float(request.form['spread1'])
        spread2 = float(request.form['spread2'])
        D2 = float(request.form['D2'])
        PPE = float(request.form['PPE'])

        # Make predictions using the Parkinson's disease model
        features = [MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Jitter_Abs, RAP, PPQ, DDP,
                    MDVP_Shimmer, shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        prediction = parkinsons_model.predict([features])[0]

        if prediction == 0:
            result = "The person is not likely to have Parkinson's disease."
        else:
            result = "The person is likely to have Parkinson's disease."

        return render_template('parkinsons.html', result=result)

    return render_template('parkinsons.html', result=None)



if __name__ == '__main__':
    app.run(debug=True)
