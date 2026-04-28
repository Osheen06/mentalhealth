import streamlit as st
import subprocess

st.title("Mental Health Sentiment Analyzer")
st.write("Click the button below to run the analysis.")

if st.button("Run Analysis"):
    with st.spinner("Running NLP Model..."):
        # This runs your script and shows the output
        result = subprocess.run(["python3", "mental_health_sentiment_analyzer.py"], capture_output=True, text=True)
        st.success("Analysis Complete!")
        st.code(result.stdout)

