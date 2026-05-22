import streamlit as st
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoConfig
)
from pyvi.ViTokenizer import tokenize

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

VIGO_LABELS = {
    0: "amusement",
    1: "excitement",
    2: "joy",
    3: "love",
    4: "desire",
    5: "optimism",
    6: "caring",
    7: "pride",
    8: "admiration",
    9: "gratitude",
    10: "relief",
    11: "approval",
    12: "realization",
    13: "surprise",
    14: "curiosity",
    15: "confusion",
    16: "fear",
    17: "nervousness",
    18: "remorse",
    19: "embarrassment",
    20: "disappointment",
    21: "sadness",
    22: "grief",
    23: "disgust",
    24: "anger",
    25: "annoyance",
    26: "disapproval",
    27: "neutral"
}

# =========================================================
# MODEL CLASS FOR VIGO
# =========================================================

class ModelSentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        bert_model = "uitnlp/visobert"

        config = AutoConfig.from_pretrained(
            bert_model,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )

        self.bert = AutoModel.from_pretrained(
            bert_model,
            config=config
        )

        self.drop = nn.Dropout(p=0.2)

        self.fc = nn.Linear(
            self.bert.config.hidden_size,
            n_classes
        )

    def forward(self, input_ids, attention_mask):

        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        x = self.drop(output)
        x = self.fc(x)

        return {"logits": x}

# =========================================================
# TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    "uitnlp/visobert",
    use_fast=False
)

# =========================================================
# LOAD MODELS
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


@st.cache_resource
def load_vigo_model(model_path):

    model = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=False
    ) 

    model.to(DEVICE)
    model.eval()

    return model

# =========================================================
# PREDICT FUNCTIONS
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


def predict_vigo(model, text, threshold=0.5):

    text = tokenize(text)

    encoding = tokenizer(
        text,
        truncation=True,
        add_special_tokens=True,
        max_length=200,
        padding='max_length',
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

        probs = torch.sigmoid(outputs["logits"])[0]

        preds = (probs >= threshold).int()

    result = []

    for i, value in enumerate(preds):

        if value == 1:

            result.append({
                "label": VIGO_LABELS[i],
                "score": float(probs[i])
            })

    return result

# =========================================================
# UI
# =========================================================

st.title("Vietnamese Sentiment Analysis")

st.write("Select model and enter Vietnamese text")

model_option = st.selectbox(
    "Choose model",
    [
        "UIT-VSFC (3 labels)",
        "ViGoEmotions (28 labels)"
    ]
)

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

    # =====================================================
    # UIT-VSFC
    # =====================================================

    if model_option == "UIT-VSFC (3 labels)":

        with st.spinner("Loading model..."):

            model = load_uit_model("best_model.pth")

        result = predict_uit(model, text_input)

        st.success(f"Prediction: {result}")

    # =====================================================
    # VIGO
    # =====================================================

    else:

        with st.spinner("Loading model..."):

            model = load_vigo_model("visobert.pth")

        results = predict_vigo(model, text_input)

        st.success("Predicted Labels")

        if len(results) == 0:

            st.write("No emotion detected")

        else:

            for item in results:

                st.write(
                    f"• {item['label']} "
                    f"({item['score']:.4f})"
                )