from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import scipy.stats as st

app = Flask(__name__)

# Create a sample CSV file (if it doesn't exist)
def create_sample_csv():
    n_samples = 10000
    data = pd.DataFrame({
        "age": np.random.randint(18, 80, n_samples),
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "job_type": np.random.choice([ "Unemployed", "Office", "Labor", "Self-Employed", "Architect", "Banker", "Bartender",
                                        "Language Translator", "Truck Driver", "Veterinarian", "Waiter/Waitress", "Writer"], n_samples),
        "income": np.random.randint(100, 10000, n_samples),
        "smoking": np.random.choice([0, 1], n_samples),
        "alcohol": np.random.choice([0, 1], n_samples),
        "drug_use": np.random.choice([0, 1], n_samples),
        "sleep_hours": np.random.randint(3, 10, n_samples),
        "exercise": np.random.choice([0, 1], n_samples),
        "family_history": np.random.choice([0, 1], n_samples),
        "stress_level": np.random.randint(1, 10, n_samples),
        "social_support": np.random.choice([0, 1], n_samples),
        "diet_quality": np.random.randint(1, 10, n_samples),
        "mental_illness": np.random.choice([0, 1], n_samples),  # Target variable
        "anxiety": np.random.choice([0, 1], n_samples),
        "depression": np.random.choice([0, 1], n_samples),
        "isolation": np.random.choice([0, 1], n_samples),
        "chronic_illness": np.random.choice([0, 1], n_samples)
    })
    data.to_csv("mental_health_data.csv", index=False)

# Read data from CSV
def load_data():
    try:
        data = pd.read_csv("mental_health_data.csv")
    except FileNotFoundError:
        create_sample_csv()
        data = pd.read_csv("mental_health_data.csv")
    return data

# Load and preprocess data
data = load_data()
data = pd.get_dummies(data, drop_first=True)

# Splitting data
X = data.drop("mental_illness", axis=1)
y = data["mental_illness"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Age group analysis
data['age_group'] = pd.cut(data['age'], bins=[18, 30, 50, 80], labels=['18-30', '31-50', '51-80'])
age_group_analysis = data.groupby('age_group')['mental_illness'].mean()

# Grouping significant and insignificant factors
significant_factors = feature_importance[feature_importance['Importance'] > feature_importance['Importance'].mean()]
insignificant_factors = feature_importance[feature_importance['Importance'] <= feature_importance['Importance'].mean()]

# Analyzing symptoms contribution
symptoms = ['anxiety', 'depression', 'isolation', 'chronic_illness']
symptom_importance = feature_importance[feature_importance['Feature'].isin(symptoms)]

# Grouping significant and insignificant symptoms
significant_symptoms = symptom_importance[symptom_importance['Importance'] > symptom_importance['Importance'].mean()]
insignificant_symptoms = symptom_importance[symptom_importance['Importance'] <= symptom_importance['Importance'].mean()]

# Lifestyle factors analysis
lifestyle_factors = ['smoking', 'alcohol', 'diet_quality']  # Adjusted to match your csv
lifestyle_importance = feature_importance[feature_importance['Feature'].isin(lifestyle_factors)]

# Calculate age confidence interval
age_mean = data['age'].mean()
age_std = data['age'].std()
age_ci = st.t.interval(0.95, len(data['age']) - 1, loc=age_mean, scale=age_std / np.sqrt(len(data['age'])))

@app.route("/")
def index():
    # Get top factors and symptoms
    top_symptom = significant_symptoms.iloc[0]['Feature'] if not significant_symptoms.empty else "N/A"
    second_top_symptom = significant_symptoms.iloc[1]['Feature'] if len(significant_symptoms) > 1 else "N/A"
    least_symptom = insignificant_symptoms.iloc[0]['Feature'] if not insignificant_symptoms.empty else "N/A"
    second_least_symptom = insignificant_symptoms.iloc[1]['Feature'] if len(insignificant_symptoms) > 1 else "N/A"
    top_factor = significant_factors.iloc[0]['Feature'] if not significant_factors.empty else "N/A"
    second_top_factor = significant_factors.iloc[1]['Feature'] if len(significant_factors) > 1 else "N/A"
    least_factor = insignificant_factors.iloc[0]['Feature'] if not insignificant_factors.empty else "N/A"
    second_least_factor = insignificant_factors.iloc[1]['Feature'] if len(insignificant_factors) > 1 else "N/A"
    top_age_group = age_group_analysis.idxmax() if not age_group_analysis.empty else "N/A"
    second_top_age_group = age_group_analysis.sort_values(ascending=False).index[1] if len(age_group_analysis) > 1 else "N/A"
    top_lifestyle = lifestyle_importance.iloc[0]['Feature'] if not lifestyle_importance.empty else "N/A"
    second_top_lifestyle = lifestyle_importance.iloc[1]['Feature'] if len(lifestyle_importance) > 1 else "N/A"

    return render_template('index.html',
                           report=classification_report(y_test, y_pred, output_dict=True),
                           top_factors=significant_factors.to_html(classes='table table-striped table-bordered table-hover'),
                           least_factors=insignificant_factors.tail(5).to_html(classes='table table-striped table-bordered table-hover'),
                           age_analysis=age_group_analysis.to_frame().to_html(classes='table table-striped table-bordered table-hover'),
                           significant_symptoms=significant_symptoms.to_html(classes='table table-striped table-bordered table-hover'),
                           insignificant_symptoms=insignificant_symptoms.to_html(classes='table table-striped table-bordered table-hover'),
                           lifestyle_factors=lifestyle_importance.to_html(classes='table table-striped table-bordered table-hover'),
                           symptom_importance=symptom_importance.to_html(classes='table table-striped table-bordered table-hover'),
                           data_length=len(data),
                           age_ci=age_ci,
                           top_symptom=top_symptom,
                           second_top_symptom=second_top_symptom,
                           least_symptom=least_symptom,
                           second_least_symptom=second_least_symptom,
                           top_factor=top_factor,
                           second_top_factor=second_top_factor,
                           least_factor=least_factor,
                           second_least_factor=second_least_factor,
                           top_age_group=top_age_group,
                           second_top_age_group=second_top_age_group,
                           top_lifestyle=top_lifestyle,
                           second_top_lifestyle=second_top_lifestyle
                           )

if __name__ == "__main__":
    app.run(debug=True)