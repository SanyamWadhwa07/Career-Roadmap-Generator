import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

# Directories
os.makedirs("models", exist_ok=True)
os.makedirs("encoders", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("===== Tech Domain Learner =====")

# Load dataset
df = pd.read_csv("../data/augmented_dataset.csv")
print(f"Original dataset shape: {df.shape}")
print(f"Columns: {', '.join(df.columns)}")

# Data exploration
print("\nValue counts for domains:")
print(df['Domain'].value_counts())
print("\nValue counts for levels:")
print(df['Level'].value_counts())

# Data cleaning
df = df.dropna(axis=1, how="all")  # Drop columns that are all NA
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Convert Month to numeric (fixing the "no of months" typo)
df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
# Handle potential missing values with median instead of dropping rows
df["Month"] = df["Month"].fillna(df["Month"].median())
print(f"\nAfter cleaning: {df.shape}")

# Create feature combinations to enrich the feature space
print("Creating feature combinations...")
df["Domain_Level"] = df["Domain"] + "_" + df["Level"]  
df["Domain_Month"] = df["Domain"] + "_" + df["Month"].astype(str)

# Encode categorical variables
label_encoder_domain = LabelEncoder()
label_encoder_level = LabelEncoder()
label_encoder_domain_level = LabelEncoder()
label_encoder_domain_month = LabelEncoder()

df["domain_encoded"] = label_encoder_domain.fit_transform(df["Domain"])
df["level_encoded"] = label_encoder_level.fit_transform(df["Level"])
df["domain_level_encoded"] = label_encoder_domain_level.fit_transform(df["Domain_Level"])
df["domain_month_encoded"] = label_encoder_domain_month.fit_transform(df["Domain_Month"])

# Print feature information
print(f"\nNumber of unique domains: {len(label_encoder_domain.classes_)}")
print(f"Number of unique levels: {len(label_encoder_level.classes_)}")
print(f"Number of unique domain-level combinations: {len(label_encoder_domain_level.classes_)}")

# Save encoders
joblib.dump(label_encoder_domain, "encoders/label_encoder_domain.pkl")
joblib.dump(label_encoder_level, "encoders/label_encoder_level.pkl")
joblib.dump(label_encoder_domain_level, "encoders/label_encoder_domain_level.pkl")
joblib.dump(label_encoder_domain_month, "encoders/label_encoder_domain_month.pkl")

# Load SentenceTransformer - using a more robust model for better embeddings
print("\nGenerating embeddings with SentenceTransformer...")
model = SentenceTransformer("all-mpnet-base-v2")  # More powerful model than MiniLM

# Fill NaNs with placeholder
df["Skill"] = df["Skill"].fillna("Unknown")
df["Recommended Resource 1"] = df["Recommended Resource 1"].fillna("Unknown")
df["Recommended Resource 2"] = df["Recommended Resource 2"].fillna("Unknown")

# Get unique items and create embeddings
unique_skills = sorted(df["Skill"].unique())
unique_res1 = sorted(df["Recommended Resource 1"].unique())
unique_res2 = sorted(df["Recommended Resource 2"].unique())

print(f"Number of unique skills: {len(unique_skills)}")
print(f"Number of unique resource 1: {len(unique_res1)}")
print(f"Number of unique resource 2: {len(unique_res2)}")

# Analyze the data imbalance
skill_counts = df["Skill"].value_counts()
print(f"Skills appearing only once: {sum(skill_counts == 1)} ({sum(skill_counts == 1)/len(skill_counts):.1%})")
res1_counts = df["Recommended Resource 1"].value_counts()
print(f"Resource 1 appearing only once: {sum(res1_counts == 1)} ({sum(res1_counts == 1)/len(res1_counts):.1%})")
res2_counts = df["Recommended Resource 2"].value_counts()
print(f"Resource 2 appearing only once: {sum(res2_counts == 1)} ({sum(res2_counts == 1)/len(res2_counts):.1%})")

# Embed full skill/resource lists
print("Creating embeddings...")
skill_embeddings = model.encode(unique_skills)
res1_embeddings = model.encode(unique_res1)
res2_embeddings = model.encode(unique_res2)

# Save texts and embeddings
joblib.dump(unique_skills, "models/unique_skills.pkl")
joblib.dump(unique_res1, "models/unique_res1.pkl")
joblib.dump(unique_res2, "models/unique_res2.pkl")

np.save("models/skill_embeddings.npy", skill_embeddings)
np.save("models/res1_embeddings.npy", res1_embeddings)
np.save("models/res2_embeddings.npy", res2_embeddings)

# Create similarity matrices for analysis
print("Computing similarity matrices to analyze embedding quality...")
skill_similarity = cosine_similarity(skill_embeddings)
res1_similarity = cosine_similarity(res1_embeddings)
res2_similarity = cosine_similarity(res2_embeddings)

# Build enhanced feature matrix X and target embeddings
print("\nPreparing feature matrix...")
X = df[["domain_encoded", "level_encoded", "Month", 
        "domain_level_encoded", "domain_month_encoded"]].values

# Scale the features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "encoders/feature_scaler.pkl")

y_skill_text = df["Skill"].tolist()
y_res1_text = df["Recommended Resource 1"].tolist()
y_res2_text = df["Recommended Resource 2"].tolist()

# Embed targets
y_skill_embed = model.encode(y_skill_text)
y_res1_embed = model.encode(y_res1_text)
y_res2_embed = model.encode(y_res2_text)

# Split data with stratified sampling if possible
print("Splitting data for training and evaluation...")
X_train, X_test, y_train_skill, y_test_skill = train_test_split(
    X_scaled, y_skill_embed, test_size=0.2, random_state=42)
_, _, y_train_text_skill, y_test_text_skill = train_test_split(
    X_scaled, y_skill_text, test_size=0.2, random_state=42)

X_train2, X_test2, y_train_res1, y_test_res1 = train_test_split(
    X_scaled, y_res1_embed, test_size=0.2, random_state=42)
_, _, y_train_text_res1, y_test_text_res1 = train_test_split(
    X_scaled, y_res1_text, test_size=0.2, random_state=42)

X_train3, X_test3, y_train_res2, y_test_res2 = train_test_split(
    X_scaled, y_res2_embed, test_size=0.2, random_state=42)
_, _, y_train_text_res2, y_test_text_res2 = train_test_split(
    X_scaled, y_res2_text, test_size=0.2, random_state=42)

# Custom scorer for embedding similarity
def top_k_accuracy_scorer(k=5):
    def scorer(estimator, X, y_true_embed):
        preds = estimator.predict(X)
        scores = []
        
        for pred_vec, true_vec in zip(preds, y_true_embed):
            sim = cosine_similarity([pred_vec], [true_vec])[0][0]
            scores.append(sim)
            
        return np.mean(scores)
    
    return make_scorer(scorer)

# Evaluation function
def evaluate_embedding_model(model, X_test, y_true_texts, all_embeddings, all_texts, top_k=5):
    """Evaluate model predictions using cosine similarity and top-k accuracy"""
    preds = model.predict(X_test)
    correct = 0
    results = []
    similarities = []
    ranks = []

    for i, (pred_vec, true_text) in enumerate(zip(preds, y_true_texts)):
        # Calculate similarities
        sims = cosine_similarity([pred_vec], all_embeddings)[0]
        similarities.append(np.max(sims))
        
        # Get top-k predictions
        topk_idx = np.argsort(sims)[-top_k:][::-1]
        topk_texts = [all_texts[i] for i in topk_idx]
        
        # Check if true text is in top-k
        match = true_text in topk_texts
        if match:
            correct += 1
            # Find rank of true prediction
            rank = topk_texts.index(true_text) + 1
            ranks.append(rank)
        
        # Store prediction results
        results.append({
            'true': true_text,
            'predictions': topk_texts,
            'correct': match,
            'max_sim': np.max(sims)
        })

    accuracy = correct / len(y_true_texts)
    
    # Additional metrics
    avg_sim = np.mean(similarities)
    mrr = np.mean([1/r for r in ranks]) if ranks else 0  # Mean Reciprocal Rank
    
    print(f"Top-{top_k} Accuracy: {accuracy:.4f}")
    print(f"Average Max Similarity: {avg_sim:.4f}")
    print(f"Mean Reciprocal Rank: {mrr:.4f}")
    print(f"Total correct: {correct}/{len(y_true_texts)}")
    
    if ranks:
        rank_counts = {i: ranks.count(i) for i in range(1, top_k+1)}
        print(f"Rank distribution: {rank_counts}")
    
    return accuracy, results, avg_sim, mrr

# Train models with advanced hyperparameter tuning
def train_model_with_advanced_tuning(X_train, y_train, X_test, y_test_text, all_embeddings, all_texts, name="Skill"):
    """Train with more extensive RandomizedSearchCV and improved param ranges"""
    print(f"\n===== Training {name} Model =====")
    
    # Larger parameter space with RandomizedSearchCV for efficiency
    param_dist = {
        'n_estimators': randint(50, 500),
        'max_depth': [None] + list(randint(5, 50).rvs(5)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None] + list(uniform(0.1, 0.9).rvs(3))
    }
    
    # Use a basic model first for baseline
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    
    print("Evaluating baseline model...")
    base_acc, _, base_sim, base_mrr = evaluate_embedding_model(
        base_model, X_test, y_test_text, all_embeddings, all_texts
    )
    
    print("\nRunning advanced hyperparameter search...")
    # Use RandomizedSearchCV to explore more parameters efficiently
    rf_random = RandomizedSearchCV(
        RandomForestRegressor(random_state=42), 
        param_distributions=param_dist,
        n_iter=25,  # Try 25 combinations
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    rf_random.fit(X_train, y_train)
    print(f"Best parameters: {rf_random.best_params_}")
    
    print("Evaluating tuned model...")
    tuned_acc, results, tuned_sim, tuned_mrr = evaluate_embedding_model(
        rf_random.best_estimator_, X_test, y_test_text, all_embeddings, all_texts
    )
    
    # Choose the best model
    if tuned_acc >= base_acc:
        print(f"Using tuned model (accuracy improved by {tuned_acc - base_acc:.4f})")
        best_model = rf_random.best_estimator_
        best_acc = tuned_acc
    else:
        print(f"Using baseline model (tuned model did not improve accuracy)")
        best_model = base_model
        best_acc = base_acc
    
    return best_model, best_acc, results

# Train all models
print("\nTraining models with advanced tuning...")
skill_model, skill_acc, skill_results = train_model_with_advanced_tuning(
    X_train, y_train_skill, X_test, y_test_text_skill, 
    skill_embeddings, unique_skills, "Skill"
)

res1_model, res1_acc, res1_results = train_model_with_advanced_tuning(
    X_train2, y_train_res1, X_test2, y_test_text_res1,
    res1_embeddings, unique_res1, "Resource 1"
)

res2_model, res2_acc, res2_results = train_model_with_advanced_tuning(
    X_train3, y_train_res2, X_test3, y_test_text_res2,
    res2_embeddings, unique_res2, "Resource 2"
)

# Function to calculate Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(y_true, y_pred):
    ranks = []
    for true_vec, pred_vec in zip(y_true, y_pred):
        # Compute cosine similarity between the true vector and predicted vectors
        similarities = cosine_similarity([true_vec], [pred_vec])[0]
        # Rank the similarities in descending order
        sorted_indices = np.argsort(similarities)[::-1]
        # Find the rank of the true vector
        rank = np.where(sorted_indices == 0)[0]
        if len(rank) > 0:
            ranks.append(1 / (rank[0] + 1))
        else:
            ranks.append(0)
    return np.mean(ranks)

# Updated top_k_accuracy to handle embedding comparisons using cosine similarity
def top_k_accuracy(y_true, y_pred, all_embeddings, k=5):
    correct = 0
    for true_vec, pred_vec in zip(y_true, y_pred):
        # Compute cosine similarity between the predicted vector and all true vectors
        similarities = cosine_similarity([pred_vec], all_embeddings)[0]
        # Get the indices of the top-k most similar embeddings
        top_k_indices = similarities.argsort()[-k:][::-1]
        # Check if the true vector is among the top-k
        if any((true_vec == all_embeddings[i]).all() for i in top_k_indices):
            correct += 1
    return correct / len(y_true)

# Updated evaluation for KNN and Decision Tree models to calculate only MRR and similarity
for model_name, model in zip(["KNN", "Decision Tree"], [KNeighborsRegressor(n_neighbors=5), DecisionTreeRegressor(max_depth=5, random_state=42)]):
    print(f"===== Evaluating {model_name} Model =====")

    # Train the model
    model.fit(X_train2, y_train_res1)

    # Predictions
    y_pred = model.predict(X_test2)

    # MRR
    mrr = mean_reciprocal_rank(y_test_res1, y_pred)
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    # Similarity
    similarities = [cosine_similarity([pred], [true])[0][0] for pred, true in zip(y_pred, y_test_res1)]
    avg_similarity = np.mean(similarities)
    print(f"Average Cosine Similarity: {avg_similarity:.4f}")

# Save models
joblib.dump(skill_model, "models/skill_model_advanced.pkl")
joblib.dump(res1_model, "models/res1_model_advanced.pkl")
joblib.dump(res2_model, "models/res2_model_advanced.pkl")

# Analyze feature importance for the skill model
feature_names = ["Domain", "Level", "Month", "Domain_Level", "Domain_Month"]
importances = skill_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance for Skill Prediction')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
print("Saved feature importance plot to plots/feature_importance.png")

# Create a prediction function for the inference pipeline
def predict_recommendations(domain, level, month_num, top_k=3):
    """Make predictions for a new user based on domain, level, and months"""
    # Load encoders
    domain_encoder = joblib.load("encoders/label_encoder_domain.pkl")
    level_encoder = joblib.load("encoders/label_encoder_level.pkl")
    domain_level_encoder = joblib.load("encoders/label_encoder_domain_level.pkl")
    domain_month_encoder = joblib.load("encoders/label_encoder_domain_month.pkl")
    scaler = joblib.load("encoders/feature_scaler.pkl")

    # Create feature combinations
    domain_level = f"{domain}_{level}"
    domain_month = f"{domain}_{str(month_num)}"

    # Encode inputs
    try:
        domain_code = domain_encoder.transform([domain])[0]
        level_code = level_encoder.transform([level])[0]
        domain_level_code = domain_level_encoder.transform([domain_level])[0]
        domain_month_code = domain_month_encoder.transform([domain_month])[0]
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Available domains: {domain_encoder.classes_}")
        print(f"Available levels: {level_encoder.classes_}")
        return None

    # Create and scale input array
    X_new = np.array([[domain_code, level_code, month_num, domain_level_code, domain_month_code]])
    X_new_scaled = scaler.transform(X_new)

    # Load models and embeddings
    skill_model = joblib.load("models/skill_model_advanced.pkl")
    res1_model = joblib.load("models/res1_model_advanced.pkl")
    res2_model = joblib.load("models/res2_model_advanced.pkl")

    skill_embed = np.load("models/skill_embeddings.npy") 
    res1_embed = np.load("models/res1_embeddings.npy")
    res2_embed = np.load("models/res2_embeddings.npy")

    unique_skills = joblib.load("models/unique_skills.pkl")
    unique_res1 = joblib.load("models/unique_res1.pkl")
    unique_res2 = joblib.load("models/unique_res2.pkl")

    # Make predictions
    skill_pred = skill_model.predict(X_new_scaled)[0]
    res1_pred = res1_model.predict(X_new_scaled)[0]
    res2_pred = res2_model.predict(X_new_scaled)[0]

    # Find most similar items
    skill_sims = cosine_similarity([skill_pred], skill_embed)[0]
    res1_sims = cosine_similarity([res1_pred], res1_embed)[0]
    res2_sims = cosine_similarity([res2_pred], res2_embed)[0]

    # Get top-k recommendations with similarity scores
    top_skills_idx = np.argsort(skill_sims)[-top_k:][::-1]
    top_res1_idx = np.argsort(res1_sims)[-top_k:][::-1]
    top_res2_idx = np.argsort(res2_sims)[-top_k:][::-1]

    top_skills = [(unique_skills[i], skill_sims[i]) for i in top_skills_idx]
    top_res1 = [(unique_res1[i], res1_sims[i]) for i in top_res1_idx]
    top_res2 = [(unique_res2[i], res2_sims[i]) for i in top_res2_idx]

    # Suggest projects based on the top skill
    top_skill = unique_skills[top_skills_idx[0]]
    project_suggestions = df[df['Skill'] == top_skill]['Project'].unique().tolist()

    return {
        "skills": top_skills,
        "resource1": top_res1,
        "resource2": top_res2,
        "projects": project_suggestions
    }

# Print final performance summary
print("\n===== Final Performance Summary =====")
print(f"Skill Prediction Accuracy: {skill_acc:.4f}")
print(f"Resource 1 Prediction Accuracy: {res1_acc:.4f}")
print(f"Resource 2 Prediction Accuracy: {res2_acc:.4f}")
print("====================================")

# Example usage of the prediction function
print("\nExample predictions:")

example_inputs = [
    ("Machine Learning", "Intermediate", 4),
    ("Web Development", "Beginner", 2),
    ("Data Analyst", "Advanced", 6)
]

for domain, level, month in example_inputs:
    print(f"\nInput: Domain='{domain}', Level='{level}', Month={month}")
    try:
        recommendations = predict_recommendations(domain, level, month)
        if recommendations:
            print("\nRecommended Skills:")
            for i, (skill, sim) in enumerate(recommendations["skills"], 1):
                print(f"  {i}. {skill} (similarity: {sim:.4f})")
            
            print("\nRecommended Resource 1:")
            for i, (res, sim) in enumerate(recommendations["resource1"], 1):
                print(f"  {i}. {res} (similarity: {sim:.4f})")
            
            print("\nRecommended Resource 2:")  
            for i, (res, sim) in enumerate(recommendations["resource2"], 1):
                print(f"  {i}. {res} (similarity: {sim:.4f})")
            
            print("\nRecommended Projects:")
            for i, project in enumerate(recommendations["projects"], 1):
                print(f"  {i}. {project}")
    except Exception as e:
        print(f"Error making prediction: {e}")