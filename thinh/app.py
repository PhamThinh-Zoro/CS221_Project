import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# CẤU HÌNH BAN ĐẦU
# ==========================================
MODEL_NAME = "uitnlp/visobert"
NUM_LABELS = 3
LABELS = ["Tiêu cực (NEG)", "Trung tính (NEU)", "Tích cực (POS)"]

# Đường dẫn đến các file trọng số (hãy đảm bảo file tồn tại trong thư mục)
MODEL_PATHS = {
    "Model Fine-tuned trên VSFC": "best_model_vsfc.pth",
    "Model Fine-tuned trên VSPL2016": "best_model_vspl.pth"
}

# ==========================================
# CÁC HÀM LOAD MODEL VÀ PREDICT
# ==========================================
@st.cache_resource
def load_tokenizer():
    """Load tokenizer và cache lại để không phải load nhiều lần"""
    return AutoTokenizer.from_pretrained(MODEL_NAME)

@st.cache_resource
def load_model(model_path):
    """Load model weight dựa trên đường dẫn được chọn"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    
    try:
        # Load weights đã fine-tune
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error(f"Không tìm thấy file trọng số: {model_path}. Vui lòng đổi tên file model tương ứng và đặt cùng thư mục.")
        return None, None
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None, None

def predict(text, model, tokenizer, device):
    """Hàm dự đoán nhãn cho câu text đầu vào"""
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(logits, dim=-1).item()
        
    return pred_idx, probabilities.cpu().numpy()[0]

# ==========================================
# GIAO DIỆN STREAMLIT
# ==========================================
def main():
    st.set_page_config(page_title="Vietnamese Sentiment Analysis", page_icon="🎭")
    st.title("🎭 Phân Loại Cảm Xúc Tiếng Việt")
    st.markdown("Dự đoán cảm xúc của văn bản (Tiêu cực, Trung tính, Tích cực) sử dụng mô hình **ViSoBERT**.")

    # 1. Chọn Model
    st.sidebar.header("Cấu hình Model")
    selected_model_name = st.sidebar.selectbox(
        "Vui lòng chọn model để dự đoán:",
        list(MODEL_PATHS.keys())
    )
    
    # 2. Khởi tạo model và tokenizer
    tokenizer = load_tokenizer()
    model_path = MODEL_PATHS[selected_model_name]
    
    with st.spinner('Đang tải mô hình... (Có thể mất một chút thời gian)'):
        model, device = load_model(model_path)

    # 3. Input Text
    st.subheader("Nhập văn bản cần dự đoán")
    user_input = st.text_area("Văn bản:", height=150, placeholder="Nhập một câu tiếng Việt vào đây để kiểm tra cảm xúc...")

    # 4. Nút Predict
    if st.button("Dự đoán 🚀"):
        if not user_input.strip():
            st.warning("Vui lòng nhập văn bản trước khi dự đoán!")
        elif model is None:
            st.error("Mô hình chưa được tải thành công. Vui lòng kiểm tra lại file .pth!")
        else:
            with st.spinner('Đang xử lý...'):
                pred_idx, probs = predict(user_input, model, tokenizer, device)
                
                predicted_label = LABELS[pred_idx]
                
                # Hiển thị kết quả
                st.success(f"**Kết quả dự đoán:** {predicted_label}")
                
                # Hiển thị độ tin cậy (Confidence scores)
                st.markdown("### Độ tin cậy (Confidence Scores):")
                col1, col2, col3 = st.columns(3)
                col1.metric(label=LABELS[0], value=f"{probs[0]*100:.2f}%")
                col2.metric(label=LABELS[1], value=f"{probs[1]*100:.2f}%")
                col3.metric(label=LABELS[2], value=f"{probs[2]*100:.2f}%")

if __name__ == "__main__":
    main()