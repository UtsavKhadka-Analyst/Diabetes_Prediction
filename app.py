import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Sidebar for Patient Input ---
st.sidebar.title("🧬 Enter Patient Details")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0)
glucose = st.sidebar.number_input("Glucose", min_value=0)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0)
insulin = st.sidebar.number_input("Insulin", min_value=0)
bmi = st.sidebar.number_input("BMI", min_value=0.0, format="%.2f")
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.2f")
age = st.sidebar.number_input("Age", min_value=0)

# --- App Title ---
st.title("🏥 Diabetes Prediction Dashboard")

# --- Load Dataset ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/UtsavKhadka-Analyst/Diabetes_Prediction/refs/heads/main/diabetes.csv"
    return pd.read_csv(url)

df = load_data()

# --- Data Cleaning ---
def remove_outliers_iqr(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

df_cleaned = remove_outliers_iqr(df)

# --- Preprocessing ---
X = df_cleaned.drop('Outcome', axis=1)
y = df_cleaned['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# --- Train Models ---
log_reg = LogisticRegression().fit(X_train, y_train)
rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
xgb = XGBClassifier(eval_metric='logloss', random_state=42).fit(X_train, y_train)
lgbm_params = {
    'min_gain_to_split': 0.0,
    'min_data_in_leaf': 10,
    'max_depth': 6,
    'num_leaves': 31,
    'verbosity': -1,  # Suppress logs
    'random_state': 42
}
lgbm = LGBMClassifier(**lgbm_params).fit(X_train, y_train)
svm = SVC(kernel='rbf', probability=True).fit(X_train, y_train)

model_names = ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'SVM']
models = [log_reg, rf, xgb, lgbm, svm]
accuracies = [accuracy_score(y_test, model.predict(X_test)) for model in models]

# --- Tabs Layout ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Exploration", "🔧 Model Training", "📈 Model Comparison", "🩺 Prediction"])

# --- Tab 1: Data Exploration ---
with tab1:
    st.subheader("🔍 Raw Dataset")
    st.dataframe(df.head())

    st.subheader("📊 Correlation Heatmap")
    fig_corr, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_cleaned.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig_corr)
    plt.close(fig)


    # --- Boxplots Before Cleaning ---
    st.subheader("📦 Boxplots (Before Outlier Removal)")
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f'Boxplot of {col}')
        st.pyplot(fig)
        plt.close(fig)


    # --- Boxplots After Cleaning ---
    st.subheader("📦 Boxplots (After Outlier Removal)")
    for col in df_cleaned.select_dtypes(include=['float64', 'int64']).columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=df_cleaned[col], ax=ax)
        ax.set_title(f'Boxplot of {col}')
        st.pyplot(fig)
        plt.close(fig)



    st.subheader("📈 Feature Distributions")
    for col in df_cleaned.select_dtypes(include=['float64', 'int64']).columns:
        fig, ax = plt.subplots()
        sns.histplot(df_cleaned[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)
        plt.close(fig)


# --- Tab 2: Model Training ---
with tab2:
    st.subheader("🔧 Model Training Results")
    for name, model in zip(model_names, models):
        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.write(f"**{name} Accuracy:** {accuracy:.2f}")
        st.text(classification_report(y_test, model.predict(X_test)))

# --- Tab 3: Model Comparison ---
with tab3:
    st.subheader("📊 Accuracy Comparison")
    fig_acc, ax = plt.subplots()
    sns.barplot(x=model_names, y=accuracies, ax=ax)
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.tick_params(axis='x', labelrotation=45)
    st.pyplot(fig_acc)
    plt.close(fig)


    st.subheader("📋 Accuracy Table")
    accuracy_table = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies}).sort_values(by='Accuracy', ascending=False)
    st.dataframe(accuracy_table)

# --- Tab 4: Patient Prediction ---
with tab4:
    st.subheader("🩺 Patient Diabetes Prediction")
input_df = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                          insulin, bmi, dpf, age]], columns=X.columns)
user_scaled = scaler.transform(input_df)

    prediction = log_reg.predict(user_scaled)[0]
    probability = log_reg.predict_proba(user_scaled)[0][1]

    result = "✅ No Diabetes" if prediction == 0 else "⚠️ Diabetes"
    st.metric(label="Prediction", value=result)
    st.write(f"**Probability of Diabetes:** {probability:.2f}")
    
