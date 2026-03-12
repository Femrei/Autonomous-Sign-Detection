import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Trafik Levha Tespit Sistemi", page_icon="🛑", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    h1 { color: #d32f2f; }
    .stButton>button { width: 100%; background-color: #d32f2f; color: white; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 🧠 1. BÖLÜM: MODEL YÜKLEME FONKSİYONLARI
# ==========================================

# --- YOLO MODELİNİ YÜKLE ---
@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

# --- FASTER R-CNN MODELİNİ YÜKLE ---
@st.cache_resource
def load_rcnn_model(path, num_classes=6):
    # Cihaz seçimi (GPU varsa kullan, yoksa CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 1. Boş bir Faster R-CNN modeli oluştur
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    
    # 2. Sınıf sayısını senin verisetine göre ayarla (5 Levha + 1 Arkaplan = 6)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 3. Senin eğittiğin ağırlıkları (.pth) yükle
    try:
        # Önce CPU'ya map ederek yüklemeyi dene
        checkpoint = torch.load(path, map_location=device)
        
        # Eğer checkpoint bir sözlükse (best_model.pth genelde böyledir) 'model_state_dict' anahtarını ara
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        # Eğer sadece ağırlıklarsa direkt yükle
        else:
            model.load_state_dict(checkpoint)
            
    except Exception as e:
        st.error(f"R-CNN Modeli yüklenirken hata: {e}")
        return None

    model.to(device)
    model.eval() # Test moduna al
    return model

# ==========================================
# 🎨 2. BÖLÜM: TAHMİN VE ÇİZİM FONKSİYONLARI
# ==========================================

# R-CNN Sınıf İsimleri (Senin etiket sırasına göre - Genelde 0 background'dur)
RCNN_CLASSES = ['__background__', 'dur', 'saga_donulmez', 'hiz_siniri20', 'park_edilebilir', 'park_edilemez']

def predict_rcnn(model, image, threshold):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Görüntüyü Tensor'a çevir
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).to(device)
    
    # Tahmin yap
    with torch.no_grad():
        prediction = model([img_tensor])[0]
    
    # Sonuçları işle
    img_np = np.array(image)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    
    # Eşik değerine göre çiz
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            x1, y1, x2, y2 = box.astype(int)
            class_name = RCNN_CLASSES[label] if label < len(RCNN_CLASSES) else str(label)
            
            # Kutu Çiz
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Yazı Yaz
            text = f"{class_name}: {score:.2f}"
            cv2.putText(img_cv2, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

# ==========================================
# 🖥️ 3. BÖLÜM: ARAYÜZ (SIDEBAR VE MAIN)
# ==========================================

st.sidebar.title("⚙️ Kontrol Paneli")
st.sidebar.markdown("---")

# MODEL LİSTESİ (Senin dosya isimlerin)
model_files = {
    "YOLOv11 (Yolov11)":      {"file": "model_v11.pt", "type": "yolo"},
    "YOLOv8 (Referans)":           {"file": "model_v8_batch16.pt", "type": "yolo"},
    "YOLOv8 (Batch 8)":            {"file": "model_v8_batch8.pt", "type": "yolo"},
    "YOLOv8 (SGD)":                {"file": "model_v8_sgd.pt", "type": "yolo"},
    "Faster R-CNN (Ağır Model)":   {"file": "faster_rcnn.pth", "type": "rcnn"}
}

selected_name = st.sidebar.selectbox("🧠 Model Seç:", list(model_files.keys()))
selected_info = model_files[selected_name]
model_path = selected_info["file"]
model_type = selected_info["type"]

# Dosya Kontrolü
if not os.path.exists(model_path):
    st.sidebar.error(f"❌ Dosya Yok: {model_path}")
    model = None
else:
    if model_type == "yolo":
        model = load_yolo_model(model_path)
        st.sidebar.success(f"✅ {selected_name} Hazır!")
    else:
        # R-CNN Yükleniyor...
        with st.spinner("Faster R-CNN PyTorch modeli yükleniyor..."):
            model = load_rcnn_model(model_path)
        if model:
            st.sidebar.success(f"✅ {selected_name} Hazır!")
            st.sidebar.warning("⚠️ Not: R-CNN, YOLO'ya göre daha yavaş çalışabilir.")

# Hassasiyet Ayarı
conf_threshold = st.sidebar.slider("🎚️ Güven Eşiği", 0.0, 1.0, 0.45, 0.05)

# --- ANA EKRAN ---
st.title("🛑 Trafik Levhası Tespit Sistemi")

tab1, tab2 = st.tabs(["📷 Canlı Kamera", "🖼️ Fotoğraf Yükle"])

# --- TAB 1: KAMERA ---
with tab1:
    col1, col2 = st.columns([1, 4])
    run = col1.checkbox('Kamerayı Başlat')
    FRAME_WINDOW = col2.image([])
    
    if run and model:
        camera = cv2.VideoCapture(0)
        
        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Kamera açılmadı.")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MODEL TİPİNE GÖRE TAHMİN
            if model_type == "yolo":
                results = model.predict(frame_rgb, conf=conf_threshold, verbose=False)
                output_frame = results[0].plot()
            else:
                # R-CNN Tahmini (Image objesi gönderiyoruz)
                pil_image = Image.fromarray(frame_rgb)
                output_frame = predict_rcnn(model, pil_image, conf_threshold)
            
            FRAME_WINDOW.image(output_frame)
        camera.release()

# --- TAB 2: FOTOĞRAF ---
with tab2:
    uploaded_file = st.file_uploader("Fotoğraf Yükle...", type=['jpg', 'png', 'jpeg'])

    if uploaded_file and model:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📷 Orijinal Görsel**")
            st.image(image, use_container_width=True)

        if st.button("🔍 Tespit Et"):
            with st.spinner("Analiz ediliyor..."):
                if model_type == "yolo":
                    res = model.predict(image, conf=conf_threshold)
                    out_img = res[0].plot()
                    # Renk düzeltme
                    if isinstance(out_img, np.ndarray) and out_img.shape[-1] == 3:
                        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
                else:
                    out_img = predict_rcnn(model, image, conf_threshold)

                st.session_state["result_img"] = out_img

        if "result_img" in st.session_state:
            with col2:
                st.markdown("**🔍 Tespit Sonucu**")
                st.image(st.session_state["result_img"], use_container_width=True)