import streamlit as st
import pandas as pd
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from lime import lime_text
import re
import joblib

def load_vect_and_model(vectorizer_path, classifier_path):
    try:
        text_vectorizer = load(vectorizer_path)
        classifier = load(classifier_path)
    except FileNotFoundError as e:
        st.error(f"File not found: {e.filename}. Please ensure the file is in the correct directory.")
        return None, None
    return text_vectorizer, classifier

# Specify the paths to your joblib files
vectorizer_path = "vectorizer.joblib"
classifier_path = "classifier.joblib"

text_vectorizer, classifier = load_vect_and_model(vectorizer_path, classifier_path)

if text_vectorizer and classifier:
    def vectorize_text(texts):
        return text_vectorizer.transform(texts)

    def pred_class(texts):
        return classifier.predict(vectorize_text(texts))

    def pred_probs(texts):
        return classifier.predict_proba(vectorize_text(texts))

    def create_colored_review(review, word_contributions):
        tokens = re.findall(r'\b\w+\b', review)
        modified_review = ""

        for token in tokens:
            if token in word_contributions["word"].values:
                idx = word_contributions["word"].values.tolist().index(token)
                contribution = word_contributions.iloc[idx]["Contribution"]
                modified_review += f'<span style="color:{"green" if contribution > 0 else "red"}">{token}</span> '
            else:
                modified_review += token + " "

        return modified_review

    explainer = lime_text.LimeTextExplainer(class_names=classifier.classes_)

    st.title("Reviews Classification (Positive vs Negative)")

    review = st.text_area(label="Enter Review Here:", value='Enjoy Dashboard', height=200)

    submit = st.button("Classify")

    if submit and review:
        prediction = pred_class([review])[0]
        probs = pred_probs([review])[0]

        st.markdown(f"**Prediction: {prediction}**")
        confidence = probs[1] * 100 if prediction == "positive" else probs[0] * 100
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

        # Explanation
        if prediction != "positive":
            explanation = explainer.explain_instance(review, classifier.predict_proba, num_features=10)
            word_contributions = pd.DataFrame(explanation.as_list(), columns=["Contribution", "word"])
            colored_review = create_colored_review(review, word_contributions)
            st.markdown(f"**Explanation:** {colored_review}")
else:
    st.error("Model or vectorizer not loaded. Please check the files and try again.")