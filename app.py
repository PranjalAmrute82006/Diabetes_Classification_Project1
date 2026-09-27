import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🩺 Diabetes Prediction System")

st.write(
    "Machine Learning based diabetes classification "
    "using Logistic Regression and Random Forest."
)

st.divider()


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("diabetes.csv")

st.subheader("📊 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Total Records",
        df.shape[0]
    )

with col2:
    st.metric(
        "Total Features",
        df.shape[1] - 1
    )

with st.expander("👀 View Dataset"):
    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df.drop("Outcome", axis=1)

y = df["Outcome"]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

lr = LogisticRegression(
    max_iter=1000
)

lr.fit(
    X_train,
    y_train
)

lr_prediction = lr.predict(
    X_test
)


# =========================================================
# RANDOM FOREST
# =========================================================

rf = RandomForestClassifier(
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

rf_prediction = rf.predict(
    X_test
)


# =========================================================
# RANDOM FOREST METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    rf_prediction
)

precision = precision_score(
    y_test,
    rf_prediction
)

recall = recall_score(
    y_test,
    rf_prediction
)

f1 = f1_score(
    y_test,
    rf_prediction
)

rf_probability_test = rf.predict_proba(
    X_test
)[:, 1]

roc_auc = roc_auc_score(
    y_test,
    rf_probability_test
)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.subheader("📊 Random Forest Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:

    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

with col3:

    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

with col4:

    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )


st.info(
    f"📈 ROC-AUC Score: {roc_auc * 100:.2f}%"
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.divider()

st.subheader("👤 Patient Information")

st.write(
    "Enter the patient's information below "
    "to predict the diabetes outcome."
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120,
        step=1
    )


with col2:

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70,
        step=1
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=25,
        step=1
    )


with col3:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=90,
        step=1
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=80.0,
        value=28.5,
        step=0.1
    )


with col4:

    pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.35,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.write("")

predict_button = st.button(
    "🔍 Predict Diabetes",
    use_container_width=True
)


# =========================================================
# PREDICTION RESULT
# =========================================================

if predict_button:

    new_patient = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            pedigree,
            age
        ]],
        columns=X.columns
    )


    # Prediction
    prediction = rf.predict(
        new_patient
    )[0]


    # Probability
    probability = rf.predict_proba(
        new_patient
    )[0][1]


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.divider()

    st.subheader("📋 Prediction Result")


    if prediction == 1:

        st.error(
            "🔴 Diabetes Predicted"
        )

        st.markdown(
            "## Prediction: Diabetic (1)"
        )

        st.write(
            "The Random Forest model classified "
            "this patient as diabetic."
        )

    else:

        st.success(
            "🟢 No Diabetes Predicted"
        )

        st.markdown(
            "## Prediction: Not Diabetic (0)"
        )

        st.write(
            "The Random Forest model classified "
            "this patient as not diabetic."
        )


    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    st.subheader("📈 Prediction Probability")

    st.write(
        f"**Diabetes Probability: "
        f"{probability * 100:.2f}%**"
    )

    st.progress(
        float(probability)
    )


# =========================================================
# MODEL COMPARISON
# =========================================================

st.divider()

st.subheader("🤖 Model Comparison")


# Logistic Regression metrics

lr_accuracy = accuracy_score(
    y_test,
    lr_prediction
)

lr_precision = precision_score(
    y_test,
    lr_prediction
)

lr_recall = recall_score(
    y_test,
    lr_prediction
)

lr_f1 = f1_score(
    y_test,
    lr_prediction
)


# Create comparison table

comparison = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Accuracy": [
        lr_accuracy * 100,
        accuracy * 100
    ],

    "Precision": [
        lr_precision * 100,
        precision * 100
    ],

    "Recall": [
        lr_recall * 100,
        recall * 100
    ],

    "F1 Score": [
        lr_f1 * 100,
        f1 * 100
    ]
})


comparison[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
] = comparison[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
].round(2)


st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

st.divider()

st.subheader("📈 Prediction Probability")

diabetes_probability = probability
no_diabetes_probability = 1 - probability

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "No Diabetes Probability",
        f"{no_diabetes_probability * 100:.2f}%"
    )

with col2:
    st.metric(
        "Diabetes Probability",
        f"{diabetes_probability * 100:.2f}%"
    )

st.write("Diabetes probability:")
st.progress(float(diabetes_probability))

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Dataset: Pima Indians Diabetes Dataset"
)

st.caption(
    "0 = No Diabetes | 1 = Diabetes"
)

st.warning(
    "⚠️ This application is for educational/project purposes "
    "and should not be used as a medical diagnosis."
)