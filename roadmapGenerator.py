import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(
    page_title="Tech Domain Learner",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean, minimal CSS
st.markdown("""
<style>
    h1 {
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .recommendation-card {
        background-color: #f5f7fa;
        border-radius: 5px;
        padding: 12px;
        margin-bottom: 10px;
        color: #000000;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        width: 100%;
    }
    /* Set text colors for the app */
    body {
        color: #000000;
    }
    .stTab {
        color: #000000;
    }
    .stTab [data-baseweb="tab"] {
        color: #000000;
    }
    .stTab [data-baseweb="tab-list"] {
        background-color: #f0f2f6;
    }
    .stTab [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    /* Make metrics more readable */
    .stMetric {
        color: #000000;
    }
    .stMetric .css-1wivap2 {
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Function to check if model files exist
def check_model_files():
    required_files = [
        "models/skill_model_advanced.pkl",
        "models/res1_model_advanced.pkl",
        "models/res2_model_advanced.pkl",
        "models/skill_embeddings.npy",
        "models/res1_embeddings.npy",
        "models/res2_embeddings.npy",
        "models/unique_skills.pkl",
        "models/unique_res1.pkl",
        "models/unique_res2.pkl",
        "encoders/label_encoder_domain.pkl",
        "encoders/label_encoder_level.pkl",
        "encoders/label_encoder_domain_level.pkl",
        "encoders/label_encoder_domain_month.pkl",
        "encoders/feature_scaler.pkl"
    ]
    
    missing_files = [file for file in required_files if not os.path.exists(file)]
    return len(missing_files) == 0, missing_files

# Prediction function adapted from original code
def predict_recommendations(domain, level, month_num, top_k=5):
    """Make predictions for a new user based on domain, level and months"""
    try:
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
            
            # Handle domain_level and domain_month combinations
            try:
                domain_level_code = domain_level_encoder.transform([domain_level])[0]
            except ValueError:
                # Use a default from the same domain if possible
                available_combos = domain_level_encoder.classes_
                domain_matches = [c for c in available_combos if c.startswith(f"{domain}_")]
                if domain_matches:
                    domain_level = domain_matches[0]
                    domain_level_code = domain_level_encoder.transform([domain_level])[0]
                else:
                    # Use the first class as default
                    domain_level = domain_level_encoder.classes_[0]
                    domain_level_code = 0
            
            try:
                domain_month_code = domain_month_encoder.transform([domain_month])[0]
            except ValueError:
                # Use a default from the same domain
                available_months = domain_month_encoder.classes_
                domain_matches = [c for c in available_months if c.startswith(f"{domain}_")]
                if domain_matches:
                    domain_month = domain_matches[0]
                    domain_month_code = domain_month_encoder.transform([domain_month])[0]
                else:
                    # Use the first class as default
                    domain_month = domain_month_encoder.classes_[0]
                    domain_month_code = 0
            
        except ValueError as e:
            st.error(f"Error encoding inputs: {e}")
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
        with st.spinner("Generating recommendations..."):
            skill_pred = skill_model.predict(X_new_scaled)[0]
            skill_sims = cosine_similarity([skill_pred], skill_embed)[0]
            top_skills_idx = np.argsort(skill_sims)[-top_k:][::-1]
            top_skills = [(unique_skills[i], skill_sims[i]) for i in top_skills_idx]
        
            res1_pred = res1_model.predict(X_new_scaled)[0]
            res2_pred = res2_model.predict(X_new_scaled)[0]
            
            res1_sims = cosine_similarity([res1_pred], res1_embed)[0]
            res2_sims = cosine_similarity([res2_pred], res2_embed)[0]
            
            top_res1_idx = np.argsort(res1_sims)[-top_k:][::-1]
            top_res2_idx = np.argsort(res2_sims)[-top_k:][::-1]
            
            top_res1 = [(unique_res1[i], res1_sims[i]) for i in top_res1_idx]
            top_res2 = [(unique_res2[i], res2_sims[i]) for i in top_res2_idx]
        
        return {
            "skills": top_skills,
            "resource1": top_res1,
            "resource2": top_res2
        }
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Function to load available domains and levels
def load_available_options():
    try:
        domain_encoder = joblib.load("encoders/label_encoder_domain.pkl")
        level_encoder = joblib.load("encoders/label_encoder_level.pkl")
        
        domains = sorted(domain_encoder.classes_)
        levels = sorted(level_encoder.classes_)
        
        return domains, levels
    except Exception as e:
        st.error(f"Error loading options: {str(e)}")
        return [], []

# Function to display recommendations - simplified
def display_recommendations(recommendations):
    if not recommendations:
        return
    
    tabs = st.tabs(["Skills", "Primary Resources", "Supplementary Resources"])
    
    with tabs[0]:
        for i, (skill, _) in enumerate(recommendations["skills"], 1):
            st.markdown(f"""<div class="recommendation-card">{i}. {skill}</div>""", unsafe_allow_html=True)
    
    with tabs[1]:
        for i, (res, _) in enumerate(recommendations["resource1"], 1):
            st.markdown(f"""<div class="recommendation-card">{i}. {res}</div>""", unsafe_allow_html=True)
    
    with tabs[2]:
        for i, (res, _) in enumerate(recommendations["resource2"], 1):
            st.markdown(f"""<div class="recommendation-card">{i}. {res}</div>""", unsafe_allow_html=True)

# Main function
def main():
    st.title("🚀 Tech Domain Learner")
    
    # Check if required model files exist
    models_exist, missing_files = check_model_files()
    
    if not models_exist:
        st.error("Required model files are missing. Please run the training script first.")
        return
    
    # Load available options
    domains, levels = load_available_options()
    
    if not domains or not levels:
        st.error("Failed to load domain and level options.")
        return
    
    # Create sidebar for inputs
    with st.sidebar:
        st.subheader("Your Tech Path")
        
        # User inputs
        domain = st.selectbox("Technical domain", domains)
        level = st.selectbox("Experience level", levels)
        month = st.slider("Months commitment", 1, 24, 6)
        num_recommendations = st.slider("Number of recommendations", 3, 10, 5)
        
        # Generate button
        generate_button = st.button("Generate Recommendations", use_container_width=True)
        
        st.divider()
        
        with st.expander("About"):
            st.write("This app recommends skills and resources for your Tech domain path based on domain, experience, and time commitment.")
    
    # Handle recommendation generation
    if generate_button:
        recommendations = predict_recommendations(domain, level, month, top_k=num_recommendations)
        
        if recommendations:
            st.success("Recommendations ready")
            
            # Simple summary
            col1, col2, col3 = st.columns(3)
            col1.metric("Domain", domain)
            col2.metric("Level", level)
            col3.metric("Time", f"{month} months")
            
            # Display recommendations
            display_recommendations(recommendations)
            
            # Option to download as CSV
            skills_df = pd.DataFrame(recommendations["skills"], columns=["Skill", "Confidence"])
            res1_df = pd.DataFrame(recommendations["resource1"], columns=["Primary Resource", "Relevance"])
            res2_df = pd.DataFrame(recommendations["resource2"], columns=["Supplementary Resource", "Relevance"])
            
            result_df = pd.concat([skills_df, res1_df, res2_df], axis=1)
            
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"Tech_roadmap_{domain}_{level}.csv",
                mime="text/csv",
            )
        else:
            st.error("Unable to generate recommendations. Please try different parameters.")

if __name__ == "__main__":
    main()