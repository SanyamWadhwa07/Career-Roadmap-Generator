# Tech Domain Roadmap Generator

## 📌 Overview
The **Teach Domain Roadmap Generator** is a personalized tool designed to create structured learning paths for individuals based on their time availability and preferred learning methods. This project integrates **Machine Learning algorithms** to provide adaptive recommendations and ensures an efficient roadmap for various domains, including:
- Machine Learning
- Web Development
- Cybersecurity
- AI & Data Science
- Software Development

The backend is implemented using **Streamlit** for interactive usability.

---

## 🚀 Features
- **Personalized Learning Paths**: Generates a unique roadmap based on user preferences.
- **Machine Learning Integration**: Uses algorithms like Decision Trees, KNN, and Random Forest for recommendations.
- **Time-Based Customization**: Adapts the roadmap based on the user's available time.
- **Multiple Learning Methods**: Supports courses, books, and projects for diverse learning styles.
- **Interactive UI**: Built with Streamlit for an intuitive user experience.
- **Dynamic Updates**: Refines recommendations as the user progresses.

---

## ⚙️ Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Streamlit
- Required Python libraries (listed in `requirements.txt`)

### Steps to Run
1. **Clone the Repository**
   ```bash
   git clone https://github.com/SanyamWadhwa07/Career-Roadmap-Generator.git
   cd Career-Roadmap-Generator

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Streamlit App**
   ```bash
   streamlit run roadmapgenerator.py
   ```
4. **Access the Web Interface**
   - Open your browser and go to `http://localhost:8501/`
   
---

## 📊 Machine Learning Algorithms Used
- **Decision Trees**: Helps in structured learning path decisions.
- **K-Nearest Neighbors (KNN)**: Suggests learning resources based on past user data.
- **Random Forest**: Enhances recommendation accuracy with multiple decision trees.

---

## 📂 Project Structure
```bash
Career-Roadmap-Generator/
│-- models/                 # ML models for adaptive learning
│-- roadmapgenerator.py      # Streamlit application backend
│-- requirements.txt        # Dependencies
│-- README.md               # Project documentation
```

---

## 🎯 Future Improvements
- Integrating **Natural Language Processing (NLP)** for better user interaction.
- Adding support for **real-time roadmap adjustments**.
- Implementing a **user progress tracking system**.

---

## 👥 Contributing
Contributions are welcome! If you'd like to improve this project, feel free to:
1. Fork the repository.
2. Create a feature branch.
3. Make necessary changes and commit.
4. Submit a Pull Request.

