import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Page Config
st.set_page_config(page_title="Heart AI Predictor", layout="wide")

st.title("❤️ Innovative IT: Heart Disease Risk Predictor")
st.write("This application uses a Machine Learning model (Random Forest) to predict heart disease risk.")

# 2. Load Data (Using a hosted version of the UCI dataset for speed)
# Copy and replace the old load_data section with this:
@st.cache_data
def load_data():
    # This is a guaranteed working link for the heart disease dataset
    url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv"
    return pd.read_csv(url)

df = load_data()

# 3. Simple Model Training
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. Sidebar Inputs for User
st.sidebar.header("Patient Input Metrics")

def user_input_features():
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex (1=M, 0=F)', [1, 0])
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.sidebar.slider('Serum Cholestoral (mg/dl)', 100, 600, 200)
    
    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol}
    # Add dummy values for features we aren't sliding for simplicity
    for col in X.columns:
        if col not in data:
            data[col] = X[col].mean()
            
    features = pd.DataFrame(data, index=[0])
    return features[X.columns] # Ensure column order matches

input_df = user_input_features()

# 5. Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction Results')
if prediction[0] == 1:
    st.error("⚠️ High Risk of Heart Disease")
else:
    st.success("✅ Low Risk of Heart Disease")

st.write(f"**Confidence Level:** {round(prediction_proba[0][prediction[0]]*100, 2)}%")
