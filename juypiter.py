import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Generating synthetic dataset
np.random.seed(42)
n_samples = 10000

data = pd.DataFrame({
    "age": np.random.randint(18, 80, n_samples),
    "gender": np.random.choice(["Male", "Female"], n_samples),
    "job_type": np.random.choice([ "Unemployed", "Office", "Labor", "Self-Employed", "Architect", "Banker", "Bartender",
         "Butcher", "Cashier", "Chef", "Cleaner", "Dentist", "Delivery Driver",
        "Fisherman", "Flight Attendant", "Judge", "Journalist", "Lawyer", "Librarian",
        "Mechanic", "Model", "Musician", "Nurse", "Painter", "Paramedic", "Photographer",
        "Police Officer", "Politician", "Professor", "Receptionist",
        "Security Guard", "Shopkeeper", "Software Developer", "Soldier",
         "Taxi Driver", "Teacher", "Technician", "Tour Guide",
        "Language Translator", "Truck Driver", "Veterinarian", "Waiter/Waitress", "Writer"], n_samples),
    "income": np.random.randint(100, 10000, n_samples),
    "education": np.random.choice(["None", "Primary", "Secondary", "Tertiary"], n_samples),
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

# Convert categorical variables to numeric
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

# Results
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nTop Contributing Factors to Mental Illness:\n", feature_importance.head(5))
print("\nLeast Contributing Factors:\n", feature_importance.tail(5))

# Age group analysis
data['age_group'] = pd.cut(data['age'], bins=[18, 30, 50, 80], labels=['18-30', '31-50', '51-80'])
age_group_analysis = data.groupby('age_group')['mental_illness'].mean()
print("\nMental Illness by Age Group:\n", age_group_analysis)

# Grouping significant and insignificant factors
significant_factors = feature_importance[feature_importance['Importance'] > feature_importance['Importance'].mean()]
insignificant_factors = feature_importance[feature_importance['Importance'] <= feature_importance['Importance'].mean()]
print("\nSignificant Contributing Variables:\n", significant_factors)
print("\nInsignificant Contributing Variables:\n", insignificant_factors)

# Analyzing symptoms contribution
symptoms = [ 'anxiety', 'depression', 'isolation', 'chronic_illness', 'Stress', 'Fatigue',
        'Insomnia', 'Irritability', 'Mood swings', 'Headaches', 'Loss of appetite',
        'Overeating', 'Weight loss', 'Weight gain', 'High blood pressure',
        'Heart palpitations', 'Shortness of breath', 'Dizziness',
        'Digestive issues (constipation, diarrhea, bloating)', 'Body aches and pains',
        'Muscle tension', 'Nausea', 'Low energy levels', 'Lack of motivation',
        'Memory problems', 'Difficulty concentrating', 'Panic attacks',
        'Sweating excessively', 'Trembling or shaking', 'Feeling overwhelmed',
        'Chest pain (not heart-related but stress-related)',
        'Numbness or tingling in hands/feet',
        'Frequent colds or infections (weakened immune system)']
symptom_importance = feature_importance[feature_importance['Feature'].isin(symptoms)]
print("\nSymptoms Contribution:\n", symptom_importance)

# Grouping significant and insignificant symptoms
significant_symptoms = symptom_importance[symptom_importance['Importance'] > symptom_importance['Importance'].mean()]
insignificant_symptoms = symptom_importance[symptom_importance['Importance'] <= symptom_importance['Importance'].mean()]
print("\nSignificant Symptoms:\n", significant_symptoms)
print("\nInsignificant Symptoms:\n", insignificant_symptoms)

# Lifestyle factors analysis
lifestyle_factors = [  'Smoking', 'Alcohol consumption', 'Drug use', 'Sleep hours', 'Exercise',
        'Diet quality', 'Living without marrying (for men)',
        'Living without marrying (for women)', 'Living in rented houses',
        'Living in high-crime areas', 'Living in dirty/unsanitary areas',
        'Stress levels', 'Work-life balance', 'Social isolation/loneliness',
        'Screen time (excessive phone/TV/computer use)',
        'Sedentary lifestyle (lack of movement)', 'Overworking (long working hours)',
        'Exposure to pollution (air, noise, water pollution)',
        'Unhealthy relationships (toxic friendships/partnerships)',
        'Spending habits (financial stress, debt)',
        'Hygiene habits (personal and environmental cleanliness)',
        'Access to healthcare', 'Eating fast food frequently', 'Skipping meals',
        'Excessive caffeine consumption', 'Gambling addiction',
        'Exposure to violence or abuse', 'Spending too much time indoors (lack of sunlight)',
        'Lack of hobbies or recreational activities', 'Excessive social media use',
        'Poor time management', 'Frequent travel (affecting routine and health)']
lifestyle_importance = feature_importance[feature_importance['Feature'].isin(lifestyle_factors)]
print("\nLifestyle Factors Contribution:\n", lifestyle_importance)

# Visualization
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance['Feature'], y=feature_importance['Importance'])
plt.xticks(rotation=90)
plt.title("Feature Importance in Predicting Mental Illness")
plt.show()
