# 🧠 Mental Health Sentiment Analyzer
### 3rd Year NLP Engineering Project

This project uses **Natural Language Processing (NLP)** to classify social media-style posts into 7 distinct mental health categories. It is designed to identify emotional states and assist in multi-class sentiment classification for clinical or research insights.

**Live Demo:** [View App on Streamlit](https://osheen06-mentalhealth-app-lmbogc.streamlit.app/)

---

## 🚀 What This Project Does
*   **Multi-Class Classification:** Uses a machine learning model to categorize text into 7 labels: *Depression, Anxiety, Bipolar Disorder, PTSD, OCD, Suicidal Ideation,* and *Neutral Wellness*.
*   **Synthetic Data Generation:** Generates a balanced dataset of ~2,700 samples for training and testing.
*   **Performance Metrics:** Achieves a high Accuracy (~99.82%) and Macro F1-Score (0.9983).
*   **NLP Pipeline:** Handles text preprocessing including tokenization, stop-word removal, and lemmatization using NLTK.

---

## 🛠 How to Run (Human Language Version)

### 1. Get the Code
First, bring the code from GitHub onto your computer:
```bash
git clone [https://github.com/Osheen06/mentalhealth.git](https://github.com/Osheen06/mentalhealth.git)
cd mentalhealth
```

### 2. Prepare Your Computer (Mac/Windows)
If you are on a Mac, you might run into "SSL Certificate" errors when downloading NLTK files. Run this first to fix it:
*   **Mac Users:** Go to your **Applications** folder → **Python 3.10** → Double-click `Install Certificates.command`.
*   **Install Dependencies:**
```bash
pip install scikit-learn scipy nltk matplotlib seaborn pandas numpy streamlit
```

### 3. Run the Analysis
You have two ways to see this in action:

**A. The "Engine Room" (Terminal):**
To see the raw data and training process:
```bash
python3 mental_health_sentiment_analyzer.py
```

**B. The "Interactive App" (Browser):**
To launch the clean, clickable interface:
```bash
python3 -m streamlit run app.py
```
*This will open a new tab in your browser where you can just click "Run Analysis" to see the charts.*

---

## 📂 Project Structure
*   `mental_health_sentiment_analyzer.py`: The core NLP logic and model training.
*   `app.py`: The web interface wrapper for Streamlit.
*   `outputs/`: Where the script saves your performance charts and graphs.
*   `requirements.txt`: A list of all the libraries needed for the project to work.

---
