import json
from pathlib import Path

import numpy as np
import streamlit as st
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

ARTIFACTS = Path("artifacts")

st.set_page_config(page_title="NewsLens | News Classification", page_icon="📰", layout="wide")

st.title("📰 NewsLens")
st.caption("AG News classification · Traditional Machine Learning vs CNN + GloVe")
st.write(
    "Enter a news headline and description. The dashboard compares a traditional "
    "TF-IDF + Naive Bayes model with the CNN using pretrained GloVe embeddings."
)

# Load model artifacts saved from the training notebook.
@st.cache_resource
def load_artifacts():
    required = [
        ARTIFACTS / "tfidf_vectorizer.joblib",
        ARTIFACTS / "tfidf_nb_classifier.joblib",
        ARTIFACTS / "glove_cnn.keras",
        ARTIFACTS / "tokenizer.json",
        ARTIFACTS / "model_info.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Model files are missing:\n- " + "\n- ".join(missing)
            + "\n\nSave the artifacts from your notebook first. See the instructions below the app."
        )

    tfidf_vectorizer = joblib.load(ARTIFACTS / "tfidf_vectorizer.joblib")
    tfidf_classifier = joblib.load(ARTIFACTS / "tfidf_nb_classifier.joblib")
    cnn_model = tf.keras.models.load_model(ARTIFACTS / "glove_cnn.keras")
    with open(ARTIFACTS / "tokenizer.json", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())
    with open(ARTIFACTS / "model_info.json", encoding="utf-8") as f:
        info = json.load(f)
    return tfidf_vectorizer, tfidf_classifier, cnn_model, tokenizer, info

CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

with st.sidebar:
    st.header("About the project")
    st.markdown("**Dataset:** AG News")
    st.markdown("**Traditional model:** TF-IDF + Multinomial Naive Bayes")
    st.markdown("**Deep learning model:** 1D CNN + fixed pretrained GloVe vectors")
    st.markdown("**Classes:** World, Sports, Business, Sci/Tech")
    st.caption("Confidence is the model's score for its selected class, not a guarantee of correctness.")

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("Enter news article")
    headline = st.text_input("Headline *", placeholder="e.g., Technology company announces a new product")
    description = st.text_area("Description *", height=160, placeholder="Write a short description of the news article...")
    classify = st.button("Classify and compare", type="primary", use_container_width=True)

with right:
    st.subheader("How it works")
    st.markdown(
        "1. The headline and description are combined.\n"
        "2. Both models process the same text using their own preprocessing.\n"
        "3. Predictions and available confidence scores are shown side by side."
    )
    st.info("Enter both fields for the most comparable result.")

if classify:
    if not headline.strip() or not description.strip():
        st.warning("Please enter both a headline and a description.")
    else:
        combined_text = (headline.strip() + " " + description.strip())
        try:
            tfidf_vectorizer, tfidf_classifier, cnn_model, tokenizer, info = load_artifacts()
            maxlen = int(info.get("maxlen", 70))

            # Traditional ML prediction
            tfidf_features = tfidf_vectorizer.transform([combined_text])
            traditional_pred = int(tfidf_classifier.predict(tfidf_features)[0])
            traditional_label = CATEGORIES[traditional_pred] if 0 <= traditional_pred < 4 else str(traditional_pred)
            traditional_conf = None
            if hasattr(tfidf_classifier, "predict_proba"):
                traditional_probs = tfidf_classifier.predict_proba(tfidf_features)[0]
                traditional_conf = float(np.max(traditional_probs))

            # CNN + GloVe prediction. Match the notebook's tokenizer and maxlen=70 padding.
            seq = tokenizer.texts_to_sequences([combined_text])
            padded = pad_sequences(seq, maxlen=maxlen)
            cnn_probs = cnn_model.predict(padded, verbose=0)[0]
            cnn_pred = int(np.argmax(cnn_probs))
            cnn_label = CATEGORIES[cnn_pred]
            cnn_conf = float(cnn_probs[cnn_pred])

            st.divider()
            st.subheader("Prediction comparison")
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.markdown("#### Traditional ML")
                st.metric("Predicted category", traditional_label)
                if traditional_conf is not None:
                    st.metric("Model confidence", f"{traditional_conf:.1%}")
                    st.progress(max(0.0, min(1.0, traditional_conf)))
                st.caption("TF-IDF + Multinomial Naive Bayes")
            with c2:
                st.markdown("#### CNN + GloVe")
                st.metric("Predicted category", cnn_label)
                st.metric("Model confidence", f"{cnn_conf:.1%}")
                st.progress(max(0.0, min(1.0, cnn_conf)))
                st.caption("1D CNN + pretrained, fixed GloVe embeddings")

            if traditional_label == cnn_label:
                st.success(f"Both models predict: **{cnn_label}**")
            else:
                st.warning(
                    f"The models disagree: Traditional ML predicts **{traditional_label}**, "
                    f"while CNN + GloVe predicts **{cnn_label}**."
                )

            with st.expander("CNN scores for all four categories"):
                for label, score in zip(CATEGORIES, cnn_probs):
                    st.write(f"**{label}** — {float(score):.1%}")
                    st.progress(max(0.0, min(1.0, float(score))))

            st.caption("Scores are model outputs and may not be calibrated probabilities.")
        except Exception as e:
            st.error(str(e))

with st.expander("Model performance (dataset test accuracy)"):
    st.write(
        "Add your verified test-set accuracy values here after rerunning the notebook "
        "with the TF-IDF prediction bug fixed. Dataset accuracy is different from the "
        "confidence score for a single article."
    )
    st.caption("Do not enter estimated or unverified accuracy values.")
