import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder   #import libarary
encoder=LabelEncoder()   

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (these must match your training feature columns)
st.sidebar.header("Input Employee Details")

# ✨ Replace these fields with your dataset's actual input columns
age = st.sidebar.slider("Age", 18, 65, 30)
worclass = st.sidebar.selectbox("Work Class",['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov',
                                            'Federal-gov', 'Self-emp-inc', 'Without-pay', 'Never-work'])
educational_num = st.sidebar.slider("Education Level (0-16)", 0, 16, 10)
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Mexico", "Philippines", "Germany", "Canada",
    "India", "Puerto-Rico", "El-Salvador", "Cuba", "England",
    "Jamaica", "South", "China", "Italy", "Dominican-Republic"
])
marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Never-married", "Divorced", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])  

# education = st.sidebar.selectbox("education", [
#     "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
# ])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
# experience = st.sidebar.slider("experience", 0, 40, 5)
gender= st.sidebar.selectbox('gender' , ['male' ,'female'] ) 
race = st.sidebar.selectbox( 'race' , ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'] )
fnlgwt = st.sidebar.slider("Final Weight", 1000, 100000, 50000)
capital_gain = st.sidebar.slider("Capital Gain", 0, 100000, 5000)
capital_loss = st.sidebar.slider("Capital Loss", 0, 5000, 1000)
relationship = st.sidebar.selectbox("Relationship", [
    "Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"
])

# Build input DataFrame (⚠️ must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [worclass],
    'fnlwgt': [fnlgwt],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': ['Not-in-family'] , # Default value, adjust as needed
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country],
    # 'education': [education],
    # 'experience': [experience],
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    for col in input_df:
        input_df[col] = encoder.fit_transform(input_df[col])

    # [input_df] = encoder.fit_transform([input_df])  # Ensure the prediction is in the correct format
    prediction = model.predict(input_df)
    
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
# !streamlit run app.py