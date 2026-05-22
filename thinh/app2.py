import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Vietnamese Sentiment Analysis",
    layout="centered"
)

# =========================================================
# LABELS
# =========================================================

UIT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# =========================================================
# TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    "uitnlp/visobert",
    use_fast=False
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_uit_model(model_path):

    model = AutoModelForSequenceClassification.from_pretrained(
        "uitnlp/visobert",
        num_labels=3
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)
    model.eval()

    return model

# =========================================================
# PREDICT FUNCTION
# =========================================================

def predict_uit(model, text):

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pred = torch.argmax(outputs.logits, dim=-1).item()

    return UIT_LABELS[pred]

# =========================================================
# UI
# =========================================================

st.title("Vietnamese Sentiment Analysis")

st.write("Phân tích cảm xúc văn bản tiếng Việt (Mô hình UIT-VSFC)")

text_input = st.text_area(
    "Input text",
    height=150,
    placeholder="Nhập câu tiếng Việt..."
)

predict_button = st.button("Predict")

# =========================================================
# INFERENCE
# =========================================================

if predict_button:

    if text_input.strip() == "":
        st.warning("Please input text")
        st.stop()

    with st.spinner("Loading model..."):
        model = load_uit_model("best_model.pth")

    result = predict_uit(model, text_input)

    st.success(f"**Prediction:** {result}")