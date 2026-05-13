import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from datetime import datetime
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.express as px

# --- 1. CẤU HÌNH GIAO DIỆN & BACKGROUND VŨ TRỤ ---
st.set_page_config(layout="wide", page_title="App Dự Đoán Mức Độ Stress")

bg_url = "https://cdn.pixabay.com/photo/2016/07/22/16/29/space-1535269_1280.jpg"

st.markdown(f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("{bg_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stMarkdown, p, li, span {{
        color: #E0E0E0 !important;
        text-shadow: 1px 1px 2px #000000;
    }}
    .stSelectbox label p {{
        color: #FFD700 !important; 
        font-weight: bold;
        font-size: 1.1rem;
        text-shadow: 2px 2px 4px #000000;
    }}
    div[data-baseweb="select"] > div {{
        background-color: rgba(20, 20, 20, 0.8) !important;
        border: 1px solid #00FFFF !important;
        color: white !important;
    }}
    ul[data-baseweb="menu"] {{
        background-color: #1A1A1A !important;
    }}
    li[role="option"] {{
        color: #FFFFFF !important;
        font-size: 1rem !important;
        padding: 10px !important;
    }}
    li[role="option"]:hover {{
        background-color: #00FFFF !important;
        color: #000000 !important;
        font-weight: bold;
    }}
    [data-testid="stDataFrame"] {{
        background-color: rgba(0, 0, 0, 0.7);
        border-radius: 10px;
        padding: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

current_dir = os.path.dirname(os.path.abspath(__file__))

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- 2. XỬ LÝ ĐIỀU HƯỚNG TRANG (ROUTING) ---
if 'page' not in st.session_state:
    st.session_state.page = "Trang Chủ"
if 'history' not in st.session_state:
    st.session_state.history = []

def change_page(page_name):
    st.session_state.page = page_name

def logout():
    st.session_state.history = []
    st.session_state.page = "Trang Chủ"
    st.rerun()

# --- 3. HUẤN LUYỆN MÔ HÌNH ---
@st.cache_resource
def train_model():
    file_path = os.path.join(current_dir, 'STRESSLEVELDATA.csv')
    if not os.path.exists(file_path):
        st.error(f"🚨 KHÔNG TÌM THẤY FILE: Hãy đảm bảo file 'STRESSLEVELDATA.csv' nằm tại: {current_dir}")
        st.stop()
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='latin1')
    
    data.columns = ['TuTin', 'GiacNgu', 'ApLucHocTap', 'LoAu', 'BatNat', 'HoTro', 'StressLevel']
    X = data[['TuTin', 'GiacNgu', 'ApLucHocTap', 'LoAu', 'BatNat', 'HoTro']]
    y = data['StressLevel']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    
    model = Perceptron(max_iter=1000, eta0=0.05, random_state=0)
    model.fit(X_train_std, y_train.values.ravel())
    return model, sc

model, scaler = train_model()

# ==========================================
# TRANG CHỦ: ĐO MỨC ĐỘ STRESS
# ==========================================
if st.session_state.page == "Trang Chủ":
    
    # Hiển thị Logo
    logo_path = os.path.join(current_dir, 'strees.jpg')
    if os.path.exists(logo_path):
        img_b64 = get_base64_of_bin_file(logo_path)
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
                <div style="display: flex; align-items: center;">
                    <img src="data:image/jpeg;base64,{img_b64}" width="70" style="border-radius: 10px; margin-right: 15px; border: 2px solid #00FFFF; box-shadow: 0 0 10px #00FFFF;">
                    <h1 style='color: #00FFFF; text-shadow: 2px 2px 4px #000000, 0 0 10px #00FFFF; margin: 0;'>Ứng Dụng Dự Đoán Tâm Lý</h1>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown("<h1 style='color: #00FFFF; text-shadow: 2px 2px 4px #000000, 0 0 10px #00FFFF;'>Ứng Dụng Dự Đoán Tâm Lý</h1>", unsafe_allow_html=True)

    # Nút chuyển trang & Đăng xuất
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 8])
    with col_nav1:
        st.button("📊 Xem Lịch Sử Đầy Đủ", on_click=change_page, args=("Lịch Sử",), use_container_width=True)
    with col_nav2:
        st.button("🚪 Đăng Xuất", on_click=logout, type="secondary", use_container_width=True)

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.header("📝 Trắc Nghiệm Tâm Lý")
        
        dict_tutin = {"Rất thấp (Thiếu tự tin trầm trọng)": 3, "Thấp (Hơi tự ti)": 8, "Dưới trung bình": 14, "Trên trung bình (Khá tự tin)": 20, "Cao (Rất tự tin)": 27}
        dict_giacngu = {"Rất kém (Mất ngủ thường xuyên)": 0, "Kém": 1, "Hơi kém": 2, "Trung bình": 3, "Tốt": 4, "Rất tốt (Ngủ sâu, đủ giấc)": 5}
        dict_apluc = {"Không có": 0, "Rất thấp": 1, "Thấp": 2, "Trung bình": 3, "Cao": 4, "Rất cao (Quá tải)": 5}
        dict_loau = dict_apluc 
        dict_batnat = dict_apluc
        dict_hotro = {"Không có ai giúp đỡ": 0, "Ít (Cảm thấy cô đơn)": 1, "Trung bình": 2, "Nhiều (Gia đình/Bạn bè luôn bên cạnh)": 3}
        
        tu_tin = st.selectbox("🎯 1. Mức độ tự tin của bạn?", list(dict_tutin.keys()))
        giac_ngu = st.selectbox("🛌 2. Chất lượng giấc ngủ gần đây?", list(dict_giacngu.keys()))
        ap_luc = st.selectbox("📚 3. Áp lực học tập/công việc?", list(dict_apluc.keys()))
        lo_au = st.selectbox("🔮 4. Mức độ lo âu về tương lai?", list(dict_loau.keys()))
        bat_nat = st.selectbox("⚠️ 5. Bạn có bị bắt nạt/cô lập không?", list(dict_batnat.keys()))
        ho_tro = st.selectbox("🤝 6. Sự hỗ trợ từ gia đình/bạn bè?", list(dict_hotro.keys()))
        
        st.markdown("<br>", unsafe_allow_html=True)
        btn_predict = st.button("🚀 KÍCH HOẠT DỰ ĐOÁN", type="primary", use_container_width=True)

    with col2:
        st.header("🔬 Kết Quả Phân Tích")
        
        if btn_predict:
            val_tutin = dict_tutin[tu_tin]
            val_giacngu = dict_giacngu[giac_ngu]
            val_apluc = dict_apluc[ap_luc]
            val_loau = dict_loau[lo_au]
            val_batnat = dict_batnat[bat_nat]
            val_hotro = dict_hotro[ho_tro]
            
            input_df = pd.DataFrame([[val_tutin, val_giacngu, val_apluc, val_loau, val_batnat, val_hotro]],
                                    columns=['TuTin', 'GiacNgu', 'ApLucHocTap', 'LoAu', 'BatNat', 'HoTro'])
            input_std = scaler.transform(input_df)
            prediction = model.predict(input_std)[0]
            
            if prediction == 0:
                stress_level = "BÌNH THƯỜNG"
                advice = "Tuyệt vời! Tâm lý của bạn đang ổn định. Hãy tiếp tục duy trì nhé!"
                color = "#00FF00" 
            elif prediction == 1:
                stress_level = "STRESS NHẸ / TRUNG BÌNH"
                advice = "Bạn đang chịu chút áp lực. Hãy dành thời gian nghe nhạc, tập thể thao để cân bằng lại."
                color = "#FFA500" 
            else:
                stress_level = "CẢNH BÁO: STRESS NẶNG"
                advice = "Báo động đỏ! Áp lực của bạn quá lớn. Hãy ngưng làm việc, nghỉ ngơi và tìm người chia sẻ ngay lập tức."
                color = "#FF3333" 
                
            st.markdown(f"""
            <div style="background-color: rgba(0,0,0,0.6); padding: 20px; border-radius: 15px; border: 2px solid {color}; text-align: center;">
                <h2 style="color: {color} !important; text-shadow: 0 0 10px {color}; margin: 0;">{stress_level}</h2>
                <p style="font-size: 1.1rem; margin-top: 15px;">{advice}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # CHỈNH SỬA: Lưu thời gian đo dưới dạng chuẩn đối tượng datetime
            current_time = datetime.now()
            
            st.session_state.history.append({
                "Thời gian đo": current_time, 
                "Kết quả số": int(prediction),
                "Tình trạng": stress_level
            })
            
        st.divider()
        st.subheader("Bảng Ghi Nhận Nhanh")
        if len(st.session_state.history) > 0:
            history_df = pd.DataFrame(st.session_state.history)
            
            # Format lại cột thời gian để hiển thị cho đẹp trong bảng
            display_df = history_df.copy()
            display_df['Thời gian đo'] = display_df['Thời gian đo'].dt.strftime('%H:%M:%S (%d/%m)')
            
            st.dataframe(display_df[['Thời gian đo', 'Tình trạng']].tail(3), use_container_width=True, hide_index=True)
        else:
            st.write("Chưa có dữ liệu.")

# ==========================================
# TRANG LỊCH SỬ: XEM BIỂU ĐỒ & ĐÁNH GIÁ
# ==========================================
elif st.session_state.page == "Lịch Sử":
    st.markdown("<h1 style='color: #00FFFF; text-shadow: 2px 2px 4px #000000, 0 0 10px #00FFFF;'>📈 Thống Kê & Đánh Giá Tâm Lý</h1>", unsafe_allow_html=True)
    
    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 8])
    with col_nav1:
        st.button("🏠 Quay Lại Trang Chủ", on_click=change_page, args=("Trang Chủ",), use_container_width=True)
    with col_nav2:
        st.button("🚪 Đăng Xuất", on_click=logout, type="secondary", use_container_width=True)

    if len(st.session_state.history) > 0:
        history_df = pd.DataFrame(st.session_state.history)
        
        if len(history_df) == 1:
            st.info("📌 Hãy thực hiện đo thêm ít nhất 1 lần nữa để hệ thống nối điểm thành đường biểu đồ.")
        
        # Vẽ biểu đồ Line Chart đường gấp khúc gốc (linear)
        fig = px.line(history_df, x='Thời gian đo', y='Kết quả số', markers=True, 
                      template="plotly_dark",
                      title="Biểu Đồ Theo Dõi Mức Độ Stress Qua Thời Gian",
                      labels={'Kết quả số': 'Mức độ Stress', 'Thời gian đo': 'Thời Gian Đo'})
        
        # Đặt lại màu vàng, bỏ phần uốn lượn (trở về đường gấp khúc thẳng)
        fig.update_traces(
            line=dict(color='#FFD700', width=4, shape='linear'), 
            marker=dict(size=12, color='#FFD700')
        )
        
        # Thiết lập trục tung, trục hoành
        fig.update_layout(
            yaxis = dict(
                tickmode = 'array',
                tickvals = [0, 1, 2],
                ticktext = ['Bình thường', 'Stress Nhẹ', 'Stress Nặng'],
                title_font=dict(size=16, color='white'),
                tickfont=dict(size=14, color='white'),
                showgrid=True, # Hiển thị đường kẻ ngang
                gridcolor='#444444'
            ),
            xaxis = dict(
                title_font=dict(size=16, color='white'),
                tickfont=dict(size=12, color='white'),
                tickformat='%H:%M:%S', # Hiện Giờ:Phút:Giây trên trục hoành
                showgrid=True, # Hiển thị đường kẻ dọc
                gridcolor='#444444'
            ),
            paper_bgcolor='rgba(0,0,0,0.6)', 
            plot_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=22, color='#00FFFF')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- PHẦN TỰ ĐỘNG ĐÁNH GIÁ MỨC ĐỘ TĂNG/GIẢM ---
        st.subheader("💡 Đánh Giá Tình Trạng Gần Nhất")
        if len(history_df) >= 2:
            last_score = history_df.iloc[-1]['Kết quả số']
            prev_score = history_df.iloc[-2]['Kết quả số']
            diff = last_score - prev_score
            
            if diff > 0:
                st.error(f"⚠️ Cảnh báo: Mức độ stress của bạn đang **TĂNG LÊN {diff} bậc** so với lần đo trước. Áp lực đang gia tăng, bạn cần tìm cách giải tỏa ngay, tránh làm việc quá sức!")
            elif diff < 0:
                st.success(f"✅ Tin vui: Mức độ stress của bạn đã **GIẢM ĐI {abs(diff)} bậc** so với lần đo trước. Phương pháp thư giãn của bạn đang rất hiệu quả, hãy tiếp tục phát huy!")
            else:
                st.info("⚖️ Mức độ stress của bạn **GIỮ NGUYÊN** không thay đổi so với lần trước. Hãy cố gắng duy trì sự cân bằng hoặc tìm cách giảm tải áp lực nếu đang ở mức cao.")
        else:
            st.write("📊 Cần ít nhất 2 lần đo để hệ thống so sánh và đưa ra đánh giá xu hướng.")
            
        st.divider()
        st.subheader("📋 Bảng Lịch Sử Chi Tiết")
        
        # Format lại cột thời gian để hiển thị cho đẹp trong bảng chi tiết
        display_df = history_df.copy()
        display_df['Thời gian đo'] = display_df['Thời gian đo'].dt.strftime('%H:%M:%S (%d/%m/%Y)')
        
        st.dataframe(display_df[['Thời gian đo', 'Tình trạng']], use_container_width=True, hide_index=True, height=300)
        
    else:
        st.warning("🌌 Chưa có dữ liệu lịch sử. Vui lòng quay lại Trang Chủ để thực hiện bài trắc nghiệm.")