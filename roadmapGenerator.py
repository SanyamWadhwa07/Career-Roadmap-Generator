import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Tech Learning Roadmap Generator",
    page_icon="🧠",
    layout="wide"
)

# Add a title and description
st.title("🧠 Personalized Tech Learning Roadmap Generator")
st.markdown("""
This app creates a customized learning roadmap based on your selected tech domain, 
skill level, and available time commitment. The roadmap includes recommended skills, 
learning resources, and projects.
""")

# Create sidebar for inputs
st.sidebar.header("📝 Your Learning Preferences")

@st.cache_resource
def load_models_and_encoders():
    # Load trained models
    models = {
        "Decision Tree": joblib.load("models/decision_tree_model.pkl"),
        "Random Forest": joblib.load("models/random_forest_model.pkl"),
        "KNN": joblib.load("models/knn_model.pkl")
    }

    # Load project and resource recommendation models
    project_model = joblib.load("models/project_recommendation_model.pkl")
    resource1_model = joblib.load("models/resource1_recommendation_model.pkl")
    resource2_model = joblib.load("models/resource2_recommendation_model.pkl")

    # Load encoders
    encoder_domain = joblib.load("encoders/encoder_domain.pkl")
    encoder_level = joblib.load("encoders/encoder_level.pkl")
    encoder_skill = joblib.load("encoders/encoder_skill.pkl")
    encoder_project = joblib.load("encoders/encoder_project.pkl")
    encoder_resource1 = joblib.load("encoders/encoder_resource1.pkl")
    encoder_resource2 = joblib.load("encoders/encoder_resource2.pkl")
    
    return models, project_model, resource1_model, resource2_model, encoder_domain, encoder_level, encoder_skill, encoder_project, encoder_resource1, encoder_resource2


try:
    # Load models and encoders
    models, project_model, resource1_model, resource2_model, encoder_domain, encoder_level, encoder_skill, encoder_project, encoder_resource1, encoder_resource2 = load_models_and_encoders()
    
    # User inputs in sidebar
    category = st.sidebar.selectbox("Select a Domain:", sorted(encoder_domain.classes_))
    difficulty = st.sidebar.selectbox("Select Skill Level:", ["Beginner", "Intermediate", "Advanced"])
    time_available = st.sidebar.slider("Time Available (months):", 1, 12, 3)
    model_choice = st.sidebar.radio("Choose Recommendation Algorithm:", list(models.keys()))

    def generate_full_roadmap(model, project_model, resource1_model, resource2_model, category, difficulty, time_months):
        roadmap = []
        
        # Encode input
        try:
            encoded_category = encoder_domain.transform([category])[0]
            encoded_difficulty = encoder_level.transform([difficulty])[0]
        except ValueError as e:
            st.error(f"Encoding error: {e}")
            return []
        
        current_month = 1
        seen_skills = set()
        max_attempts = min(time_months * 3, 36)  # Allow more attempts to find unique skills
        
        while current_month <= time_months and len(roadmap) < time_months:
            # Try up to max_attempts times to find unique skills
            for attempt in range(max_attempts):
                input_data = [[encoded_category, current_month, encoded_difficulty]]
                
                try:
                    # Predict skill
                    predicted_skill_encoded = model.predict(input_data)[0]
                    next_skill = encoder_skill.inverse_transform([predicted_skill_encoded])[0]
                    
                    # Skip duplicates but only within this loop
                    if next_skill in seen_skills:
                        # Try with a slightly different month input to get variety
                        adjusted_month = current_month + (attempt * 0.1)
                        input_data = [[encoded_category, adjusted_month, encoded_difficulty]]
                        predicted_skill_encoded = model.predict(input_data)[0]
                        next_skill = encoder_skill.inverse_transform([predicted_skill_encoded])[0]
                        
                        # If still a duplicate, continue to next attempt
                        if next_skill in seen_skills:
                            continue
                    
                    seen_skills.add(next_skill)
                    
                    # Predict resources and project
                    predicted_project = project_model.predict(input_data)[0]
                    predicted_resource1 = resource1_model.predict(input_data)[0]
                    predicted_resource2 = resource2_model.predict(input_data)[0]
                    
                    project = encoder_project.inverse_transform([predicted_project])[0]
                    resource1 = encoder_resource1.inverse_transform([predicted_resource1])[0]
                    resource2 = encoder_resource2.inverse_transform([predicted_resource2])[0]
                    
                    roadmap.append({
                        "month": current_month,
                        "skill": next_skill,
                        "resources": [resource1, resource2],
                        "project": project
                    })
                    
                    current_month += 1
                    break  # Break out of the attempt loop as we found a unique skill
                    
                except Exception as e:
                    st.error(f"Error in roadmap generation: {e}")
                    break
            
            # If we couldn't find a unique skill after max_attempts, force progress
            if current_month <= time_months and len(roadmap) < current_month:
                # Force a new skill by using a random modification to input
                import random
                random_adjust = random.uniform(0.5, 2.0)
                input_data = [[encoded_category, current_month * random_adjust, encoded_difficulty]]
                
                try:
                    predicted_skill_encoded = model.predict(input_data)[0]
                    next_skill = f"{encoder_skill.inverse_transform([predicted_skill_encoded])[0]} (Advanced)"
                    
                    predicted_project = project_model.predict(input_data)[0]
                    predicted_resource1 = resource1_model.predict(input_data)[0]
                    predicted_resource2 = resource2_model.predict(input_data)[0]
                    
                    project = encoder_project.inverse_transform([predicted_project])[0]
                    resource1 = encoder_resource1.inverse_transform([predicted_resource1])[0]
                    resource2 = encoder_resource2.inverse_transform([predicted_resource2])[0]
                    
                    roadmap.append({
                        "month": current_month,
                        "skill": next_skill,
                        "resources": [resource1, resource2],
                        "project": project
                    })
                    
                    seen_skills.add(next_skill)
                    current_month += 1
                except Exception as e:
                    st.error(f"Error in fallback roadmap generation: {e}")
                    current_month += 1  # Skip this month and try the next
        
        return roadmap
    
    # Button to generate roadmap
    if st.sidebar.button("🚀 Generate My Roadmap"):
        with st.spinner("Creating your personalized learning roadmap..."):
            roadmap = generate_full_roadmap(
                models[model_choice], 
                project_model, 
                resource1_model,
                resource2_model,
                category, 
                difficulty, 
                time_available
            )
            
            if roadmap:
                st.subheader(f"📚 Your {time_available}-Month Learning Path in {category}")
                
                # Create a progress bar
                st.progress(1.0)
                
                # Display roadmap in a more visually appealing way
                for idx, step in enumerate(roadmap, start=1):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.markdown(f"### Month {step['month']}")
                    
                    with col2:
                        st.markdown(f"### 🔹 {step['skill']}")
                        st.markdown("#### 📖 Recommended Resources:")
                        st.markdown(f"1. {step['resources'][0]}")
                        st.markdown(f"2. {step['resources'][1]}")
                        st.markdown("#### 🛠️ Project Challenge:")
                        st.markdown(f"**{step['project']}**")
                    
                    st.markdown("---")
                
                # Add a downloadable summary
                summary = "\n".join([f"Month {s['month']}: {s['skill']} - Project: {s['project']}" for s in roadmap])
                st.download_button(
                    label="📥 Download Roadmap Summary",
                    data=summary,
                    file_name=f"{category}_{difficulty}_roadmap.txt",
                    mime="text/plain"
                )
            else:
                st.warning("Unable to generate a roadmap with the current selections. Please try different inputs.")
    
    # Add extra information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 About This App")
    st.sidebar.markdown(
        "This app uses machine learning to recommend a personalized learning path "
        "based on your selections and our curated tech skill database."
    )
except Exception as e:
    st.error(f"Error loading models or encoders: {e}")
    st.info("This app requires pre-trained models to function. Please run the trainmodel.py script first to generate necessary model files.")
