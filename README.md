# 🛑 Trafik Levhası Tespit Sistemi | Traffic Sign Detection System


> <img width="640" height="640" alt="traffic_sign_detection_banner_1773316662524" src="https://github.com/user-attachments/assets/d3e62627-acca-47ee-a366-acc41053fad3" />


Bu proje, otonom sürüş güvenliği ve akıllı ulaşım sistemleri için kritik öneme sahip trafik levhalarını yüksek hassasiyetle tespit eden **derin öğrenme tabanlı** bir sistemdir. Türkiye Karayolları Genel Müdürlüğü (KGM) standartlarına uygun levhalar üzerinde, tek aşamalı (YOLO) ve iki aşamalı (Faster R-CNN) mimariler karşılaştırmalı olarak analiz edilmiştir.


><img width="1915" height="1025" alt="Ekran görüntüsü 2026-03-12 155048" src="https://github.com/user-attachments/assets/83b94b3d-9171-4329-8f76-b6cf7f0bea7b" />




><img width="1917" height="1033" alt="Ekran görüntüsü 2026-03-12 155704" src="https://github.com/user-attachments/assets/775c1827-b95e-42f9-9c65-98b239411436" />

---

## ✨ Öne Çıkan Özellikler

- 🎯 **Çoklu Model Desteği:** Yeni nesil YOLOv11n, YOLOv8n ve Faster R-CNN modelleri arasında dinamik geçiş.
- 📊 **Üstün Başarım:** YOLOv11n mimarisi ile **%99.5 mAP₅₀** seviyesinde zirve doğruluk oranı.
- 🛡️ **Robust Veri Seti:** 1750 adet özgün görselden oluşan, dengeli ve manuel etiketlenmiş özel veri seti.
- ⚙️ **Gelişmiş Augmentation:** Albumentations kütüphanesi ile zorlu hava ve ışık koşullarına karşı dayanıklılık.
- 💻 **Kullanıcı Dostu Arayüz:** Streamlit tabanlı kontrol paneli ile canlı kamera ve fotoğraf analizi.

---

## 📊 Eğitim ve Performans Analizi

Raporda gerçekleştirilen 5 farklı deneyin karşılaştırmalı sonuçları aşağıdadır:


| Model Mimarisi | Optimizer | Batch Size | mAP₅₀ (%) | Box Loss | İndir |
|---|---|---|---|---|:---:|
| **YOLOv11n** | Auto | 16 | **99.5** | 0.24 | [⬇️ İndir](https://github.com/Femrei/Autonomous---Sign--Detection-/releases/download/v1.0.0/model_v11.pt) |
| YOLOv8n (Referans) | AdamW | 16 | 99.2 | **0.22** | [⬇️ İndir](https://github.com/Femrei/Autonomous---Sign--Detection-/releases/download/v1.0.0/model_v8_batch16.pt) |
| YOLOv8n | AdamW | 8 | - | - | [⬇️ İndir](https://github.com/Femrei/Autonomous---Sign--Detection-/releases/download/v1.0.0/model_v8_batch8.pt) |
| YOLOv8n | SGD | 16 | 98.9 | 0.36 | [⬇️ İndir](https://github.com/Femrei/Autonomous---Sign--Detection-/releases/download/v1.0.0/model_v8_sgd.pt) |
| Faster R-CNN | ResNet50 | - | 94.0 | - | [⬇️ İndir](https://github.com/Femrei/Autonomous---Sign--Detection-/releases/download/v1.0.0/faster_rcnn.pth) |




> - **YOLOv11n**, eğitimin ilk 10 epoch'u içerisinde %90 doğruluk barajını aşarak en hızlı öğrenme (convergence) performansını göstermiştir.
> - **YOLOv8n**, 0.22 seviyesindeki box loss değeri ile nesne konumlandırma hatasını minimize etmede en başarılı model olmuştur.

---

## 📂 Veri Seti Detayları

Eğitim sürecinde kullanılan veri seti şu **5 ana sınıfı** kapsamaktadır:

1. Dur
2. Sağa dönülmez
3. 20 km/s hız sınırı
4. Park edilebilir
5. Park edilemez

Tüm veriler **CVAT** (Computer Vision Annotation Tool) üzerinden titizlikle etiketlenmiş ve eğitim süreci **NVIDIA T4 GPU** donanımı kullanılarak **Google Colab** ortamında tamamlanmıştır.


>
>![Ekran görüntüsü_12-3-2026_16251_](https://github.com/user-attachments/assets/171b7f43-116f-4d3d-9b81-fbc947b29f34)

---

## 🚀 Kullanılan Teknolojiler

- **Dil:** Python
- **Frameworks:** Ultralytics (YOLOv8/v11), PyTorch, Torchvision
- **Görüntü İşleme:** OpenCV, PIL, Albumentations
- **Arayüz:** Streamlit

---

## 📦 Kurulum ve Kullanım

**1. Depoyu Klonlayın:**
```bash
git clone https://github.com/kullanici_adiniz/repo_adiniz.git
cd Levha_Tespit_App
```

**2. Gereksinimleri Yükleyin:**
```bash
pip install streamlit ultralytics torch torchvision opencv-python Pillow albumentations
```

**3. Uygulamayı Başlatın:**
```bash
streamlit run app5_son.py
```

---

## 👤 Hazırlayan

**Furkan Emre İnce**

Bu çalışma, derin öğrenme algoritmalarının trafik güvenliği üzerindeki potansiyelini araştırmak amacıyla geliştirilmiştir.

