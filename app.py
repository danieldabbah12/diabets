import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# יצירת דאטה סינטטי
# --------------------------------------------------

np.random.seed(42)

n = 2000

age = np.random.randint(20, 80, n)
bmi = np.random.normal(27, 5, n)
glucose = np.random.normal(100, 25, n)
blood_pressure = np.random.normal(75, 12, n)

risk_score = (
    0.03 * age +
    0.08 * bmi +
    0.04 * glucose +
    np.random.normal(0, 3, n)
)

diabetes = (risk_score > np.percentile(risk_score, 65)).astype(int)

df = pd.DataFrame({
    "Age": age,
    "BMI": bmi,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "Diabetes": diabetes
})

# --------------------------------------------------
# אימון מודל
# --------------------------------------------------

X = df[["Age", "BMI", "Glucose", "BloodPressure"]]
y = df["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --------------------------------------------------
# כותרת
# --------------------------------------------------

st.title("🩺 Diabetes Risk Predictor")

st.write(
    "מערכת הדגמה לחיזוי סיכון לסוכרת באמצעות למידת מכונה."
)

# --------------------------------------------------
# גרפים
# --------------------------------------------------

st.header("📊 Data Exploration")

st.subheader("Diabetes Distribution")

diabetes_counts = df["Diabetes"].value_counts()
st.bar_chart(diabetes_counts)

st.subheader("Average Glucose by Diabetes Status")

glucose_chart = (
    df.groupby("Diabetes")["Glucose"]
    .mean()
)

st.bar_chart(glucose_chart)

st.subheader("Average BMI by Diabetes Status")

bmi_chart = (
    df.groupby("Diabetes")["BMI"]
    .mean()
)

st.bar_chart(bmi_chart)

# --------------------------------------------------
# חיזוי אישי
# --------------------------------------------------

st.header("🤖 Personal Prediction")

age_input = st.slider(
    "Age",
    20,
    80,
    40
)

bmi_input = st.slider(
    "BMI",
    15.0,
    45.0,
    25.0
)

glucose_input = st.slider(
    "Glucose",
    60,
    200,
    100
)

bp_input = st.slider(
    "Blood Pressure",
    40,
    140,
    75
)

if st.button("Calculate Risk"):

    user_data = pd.DataFrame({
        "Age": [age_input],
        "BMI": [bmi_input],
        "Glucose": [glucose_input],
        "BloodPressure": [bp_input]
    })

    probability = model.predict_proba(user_data)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Diabetes Risk",
        f"{probability * 100:.1f}%"
    )

    if probability > 0.5:
        st.error("High Risk")
    else:
        st.success("Low Risk")

    # ----------------------------------------------
    # השוואה לאוכלוסייה
    # ----------------------------------------------

    st.subheader("Your Glucose Compared to the Population")

    percentile = (
        df["Glucose"] < glucose_input
    ).mean()

    st.progress(percentile)

    st.write(
        f"Your glucose level is higher than {percentile * 100:.0f}% of the population."
    )

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

st.header("📈 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df = importance_df.set_index("Feature")

st.bar_chart(importance_df)
