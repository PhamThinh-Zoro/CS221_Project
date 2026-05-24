import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="ViSoBERT Sentiment Demo", page_icon="")
st.title("Phân tích sắc thái bình luận với ViSoBERT")
st.write("Demo mô hình fine-tune trên tập dữ liệu UIT-VSFC.")

# ==========================================
# HÀM LOAD MODEL & TOKENIZER (DÙNG CACHE ĐỂ TRÁNH LOAD LẠI KHI REFRESH)
# ==========================================
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "uitnlp/visobert"
    
    # 1. Khởi tạo tokenizer chuẩn của ViSoBERT từ Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Khởi tạo khung kiến trúc mô hình với 3 nhãn đầu ra
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # 3. Load file trọng số .pth bạn đã tải từ Kaggle về (đặt cùng thư mục với file app.py)
    # Dùng map_location='cpu' để chạy được trên máy cá nhân không có GPU
    state_dict = torch.load("visobert_vsfc.pth", map_location=torch.device('cpu'))
    
    # 4. Đập trọng số từ file .pth vào khung mô hình
    model.load_state_dict(state_dict)
    model.eval()
    
    return tokenizer, model

# Tiến hành load mô hình
with st.spinner("Đang tải mô hình ViSoBERT... Vui lòng đợi trong giây lát."):
    tokenizer, model = load_model_and_tokenizer()

# ==========================================
# GIAO DIỆN DỰ ĐOÁN (INFERENCE INTERFACE)
# ==========================================
# Nhãn tương ứng với các đầu ra 0, 1, 2 của UIT-VSFC
target_names = {0: "Negative (Tiêu cực) 😡", 1: "Neutral (Bình thường) 😐", 2: "Positive (Tích cực) 😊"}

user_input = st.text_area("Nhập bình luận/bài viết mạng xã hội cần phân tích:", "Trường UIT học phí hợp lý mà chất lượng đào tạo tốt quá!")

if st.button("Phân tích sắc thái"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập văn bản trước khi nhấn phân tích!")
    else:
        # Tokenize văn bản thô nhập từ giao diện
        inputs = tokenizer(user_input, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        
        # Dự đoán không tính gradient để tối ưu tốc độ CPU
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
        
        # Hiển thị kết quả ra màn hình demo
        st.success(f"Kết quả dự đoán: **{target_names[prediction]}**")