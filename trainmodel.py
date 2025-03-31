import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.impute import SimpleImputer
import os
# Load dataset
df = pd.read_csv("final_dataset.csv")

# Drop empty or unnamed columns
df = df.dropna(axis=1, how='all')

# Clean up any potential issues with trailing spaces
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Initialize encoders
encoder_domain = LabelEncoder()
encoder_level = LabelEncoder()
encoder_skill = LabelEncoder()
encoder_resource1 = LabelEncoder()
encoder_resource2 = LabelEncoder()
encoder_project = LabelEncoder()

# Encode categorical columns
df["domain_encoded"] = encoder_domain.fit_transform(df["Domain"].fillna("Unknown"))
df["level_encoded"] = encoder_level.fit_transform(df["Level"].fillna("Unknown"))
df["skill_encoded"] = encoder_skill.fit_transform(df["Skill"].fillna("Unknown"))
df["resource1_encoded"] = encoder_resource1.fit_transform(df["Recommended Resource 1"].fillna("Unknown"))
df["resource2_encoded"] = encoder_resource2.fit_transform(df["Recommended Resource 2"].fillna("Unknown"))
df["project_encoded"] = encoder_project.fit_transform(df["Project"].fillna("Unknown"))

# Convert month to numeric and fill missing values with the median
df["Month"] = pd.to_numeric(df["Month"], errors='coerce').fillna(df["Month"].median())

# Features & Targets for learning the next skill
X_skill = df[["domain_encoded", "Month", "level_encoded"]]
y_skill = df["skill_encoded"]

# Handle missing values using an imputer
imputer = SimpleImputer(strategy="mean")
X_skill = pd.DataFrame(imputer.fit_transform(X_skill), columns=X_skill.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_skill, y_skill, test_size=0.2, random_state=42)

# Train different models for skill recommendation
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Evaluate model performance
dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))
print("\nModel Accuracies:")
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"KNN Accuracy: {knn_accuracy:.4f}")

# Train models for recommending projects and resources
project_model = DecisionTreeClassifier(random_state=42)
project_model.fit(X_train, df.loc[X_train.index, "project_encoded"])
resource1_model = DecisionTreeClassifier(random_state=42)
resource1_model.fit(X_train, df.loc[X_train.index, "resource1_encoded"])
resource2_model = DecisionTreeClassifier(random_state=42)
resource2_model.fit(X_train, df.loc[X_train.index, "resource2_encoded"])

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("encoders", exist_ok=True)

# Save models
joblib.dump(dt_model, "models/decision_tree_model.pkl")
joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(knn_model, "models/knn_model.pkl")
joblib.dump(project_model, "models/project_recommendation_model.pkl")
joblib.dump(resource1_model, "models/resource1_recommendation_model.pkl")
joblib.dump(resource2_model, "models/resource2_recommendation_model.pkl")

# Save encoders
joblib.dump(encoder_domain, "encoders/encoder_domain.pkl")
joblib.dump(encoder_level, "encoders/encoder_level.pkl")
joblib.dump(encoder_skill, "encoders/encoder_skill.pkl")
joblib.dump(encoder_resource1, "encoders/encoder_resource1.pkl")
joblib.dump(encoder_resource2, "encoders/encoder_resource2.pkl")
joblib.dump(encoder_project, "encoders/encoder_project.pkl")
print("\nModels and encoders saved successfully!")

# Test recommendation function
def recommend_next_step(domain, month, level):
    domain_encoded = encoder_domain.transform([domain])[0]
    level_encoded = encoder_level.transform([level])[0]
    input_features = [[domain_encoded, month, level_encoded]]
    input_features = imputer.transform(input_features)  # Apply imputation

    predicted_skill = encoder_skill.inverse_transform([dt_model.predict(input_features)[0]])[0]
    predicted_resource1 = encoder_resource1.inverse_transform([resource1_model.predict(input_features)[0]])[0]
    predicted_resource2 = encoder_resource2.inverse_transform([resource2_model.predict(input_features)[0]])[0]
    predicted_project = encoder_project.inverse_transform([project_model.predict(input_features)[0]])[0]
    
    return {
        "Skill": predicted_skill,
        "Resource1": predicted_resource1,
        "Resource2": predicted_resource2,
        "Project": predicted_project
    }

# Example recommendation
print("\nExample recommendation:")
example = recommend_next_step("Machine Learning", 1, "Beginner")
print(example)