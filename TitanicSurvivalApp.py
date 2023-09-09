import streamlit as st
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load the saved logistic regression model
model_filename = r'C:\Users\HP\LibrariesForDeployment\TitanicSurvivalModel.joblib'
lr = joblib.load(model_filename)

st.title('Titanic Survival Predictor')

# Input widgets
pclass = st.selectbox('Passenger class', [1, 2, 3])
sex = st.radio('Sex', ['Male', 'Female'])
age = st.slider('Age', 0, 100, 30)
sibsp = st.slider('Number of Siblings/Spouses Aboard', 0, 10, 0)
parch = st.slider('Number of Parents/Children Aboard', 0, 10, 0)
embarked = st.selectbox('Embarked', ['Cherbourg', 'Queenstown', 'Southampton'])
e = {'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2}
title = st.selectbox('Title', ['Mr', 'Miss', 'Mrs', 'Master', 'Royal', 'Rare'])
t = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6}
title = t.get(title)
sex_input = 1 if sex == 'Male' else 0
embarked_input = e.get(embarked)
agegroup_input = 0
if age == 0:
    agegroup_input = 1
elif 0 < age <= 5:
    agegroup_input = 2
elif 5 < age <= 13:
    agegroup_input = 5
elif 13 < age <= 18:
    agegroup_input = 4
elif 18 < age <= 24:
    agegroup_input = 6
elif 24 < age < 34:
    agegroup_input = 0
else:
    agegroup_input = 3
if st.button('Predict'):
    input_data = [[pclass, sex_input, age, sibsp, parch, embarked_input, agegroup_input, title]]
    prediction = lr.predict(input_data)[0]
    prediction_proba = lr.predict_proba(input_data)[0]

    if prediction == 1:
        st.write(f"You survived with a probability of {prediction_proba[1]:.2f}")
    else:
        s = ""
        if sex_input == 1:
            s = f"He did not survive with a probability of {prediction_proba[0]:.2f}"
        else:
            s = f"She did not survive with a probability of {prediction_proba[0]:.2f}"
        st.write(s)
if st.button('Show Feature Importance'):
    if hasattr(lr, 'coef_'):
        feature_importance = lr.coef_[0]
        feature_names = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Embarked', 'AgeGroup', 'Title']

        # Display feature importance chart
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importance)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        st.pyplot(plt)
    else:
        st.write("Feature importance scores are not available for this model.")
