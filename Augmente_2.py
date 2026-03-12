# =============================================================================
# YOLO VERİ ARTIRMA (AUGMENTATION) - SADECE PİKSEL TABANLI DÖNÜŞÜMLER
# Amaç:
# - images/train içindeki görselleri al
# - Albumentations ile renk/ışık/blur gibi piksel tabanlı augmentations uygula
# - Yeni görselleri images/train_aug içine kaydet
# - Etiketleri (labels/train) aynen kopyalayıp labels/train_aug içine yaz
#
# Not:
# - Geometrik dönüşüm (rotate, crop, flip vs.) yapılmadığı için bbox koordinatları değişmez.
# - Bu yüzden etiketi hesaplamıyoruz, doğrudan kopyalıyoruz.
# =============================================================================

import os
import shutil
import numpy as np
from PIL import Image
import albumentations as A
import cv2  # (Bu kodda doğrudan kullanılmıyor ama bazı augment'ler OpenCV altyapısı kullanabilir)

# =============================================================================
# 1) AYARLAR VE KLASÖR YOLLARI
# =============================================================================

# Ana dataset klasörü (Colab için örnek path)
BASE_DIR = '/content/dataset/YOLO_HAZIR_VERI'  # Yereldeysen burayı kendi klasör yolun yap

# Eğitim görüntü ve etiket klasörleri
IMG_TRAIN_DIR = os.path.join(BASE_DIR, 'images/train')
LBL_TRAIN_DIR = os.path.join(BASE_DIR, 'labels/train')

# Augmented çıktıların kaydedileceği klasörler (orijinalle karışmasın diye ayrı tutuluyor)
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, 'images/train_aug')
OUTPUT_LBL_DIR = os.path.join(BASE_DIR, 'labels/train_aug')

# Çıktı klasörleri yoksa oluştur (varsa hata verme)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

# =============================================================================
# 2) AUGMENTATION TANIMLARI
# =============================================================================
# Albumentations Compose:
# - Dönüşümleri sırayla uygular
# - p=... parametresi, o dönüşümün uygulanma olasılığını belirtir
#
# Önemli:
# - Burada sadece "piksel" dönüşümleri var (parlaklık/kontrast, blur, renk, gamma...)
# - Rotate/Crop gibi geometrik dönüşüm olmadığı için bbox ayarı gerekmiyor.
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),            # Rastgele parlaklık / kontrast
    A.CLAHE(p=0.2),                               # Kontrast sınırlı adaptif histogram eşitleme (detay artırır)
    A.HueSaturationValue(                         # HSV uzayında renk/saturasyon/parlaklık oynaması
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.2
    ),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),     # Bulanıklık (kamera titremesi/odak dışı gibi)
    A.RandomGamma(p=0.4),                         # Gamma değişimi (ışık koşullarını simüle eder)

    # A.RandomFog(p=0.2),                         # Sis efekti (sürüm uyumuna göre çalışabilir)
    # A.RandomRain(p=0.2),                        # Yağmur efekti
])

# =============================================================================
# 3) İŞLEM DÖNGÜSÜ (TÜM TRAIN GÖRSELLERİ ÜZERİNDE)
# =============================================================================

print(f"Veri artırma başlıyor... \nKaynak: {IMG_TRAIN_DIR}")

# Train klasöründeki görsel dosyalarını filtrele (jpg/jpeg/png)
image_files = [
    f for f in os.listdir(IMG_TRAIN_DIR)
    if f.endswith(('.jpg', '.jpeg', '.png'))
]

# Kaç adet yeni veri üretildiğini saymak için sayaç
count = 0

# Her görüntü için augmentation uygula
for filename in image_files:
    # -------------------------------------------------------------------------
    # Dosya yollarını oluştur
    # -------------------------------------------------------------------------
    img_path = os.path.join(IMG_TRAIN_DIR, filename)

    # Görsel uzantısını atıp aynı isimle .txt etiket dosyası bekliyoruz
    # örn: abc.jpg -> abc.txt
    txt_filename = filename.rsplit('.', 1)[0] + '.txt'
    txt_path = os.path.join(LBL_TRAIN_DIR, txt_filename)

    # -------------------------------------------------------------------------
    # Etiket yoksa bu görseli atla (YOLO eğitiminde bozuk veri istemeyiz)
    # -------------------------------------------------------------------------
    if not os.path.exists(txt_path):
        continue

    # -------------------------------------------------------------------------
    # 1) Görseli oku ve numpy array'e çevir
    # PIL -> numpy dönüşümü Albumentations için uygundur
    # -------------------------------------------------------------------------
    image = np.array(Image.open(img_path))

    # -------------------------------------------------------------------------
    # 2) Augmentation uygula
    # transform(image=image) => {"image": augmented_image} gibi döner
    # -------------------------------------------------------------------------
    try:
        augmented = transform(image=image)
        augmented_image = augmented["image"]

        # ---------------------------------------------------------------------
        # 3) Yeni dosya isimleri üret
        # Karışmaması için başına 'aug_' ekliyoruz
        # ---------------------------------------------------------------------
        new_filename = f"aug_{filename}"
        new_txt_filename = f"aug_{txt_filename}"

        # ---------------------------------------------------------------------
        # 4) Yeni görseli kaydet
        # augmented_image numpy array -> PIL Image -> save
        # ---------------------------------------------------------------------
        save_img_path = os.path.join(OUTPUT_IMG_DIR, new_filename)
        Image.fromarray(augmented_image).save(save_img_path)

        # ---------------------------------------------------------------------
        # 5) Etiketi kopyala
        # Geometrik dönüşüm olmadığı için bbox koordinatları değişmez.
        # Bu yüzden .txt dosyasını aynen kopyalıyoruz.
        # ---------------------------------------------------------------------
        save_txt_path = os.path.join(OUTPUT_LBL_DIR, new_txt_filename)
        shutil.copy(txt_path, save_txt_path)

        # Başarılı üretim sayısını artır
        count += 1

    except Exception as e:
        # Augmentation sırasında hata olursa görüntü ismiyle birlikte yazdır
        print(f"Hata oluştu ({filename}): {e}")

# =============================================================================
# 4) ÖZET / SONUÇ BİLGİSİ
# =============================================================================
print(f"\n[İŞLEM TAMAM] Toplam {count} adet yeni veri üretildi.")
print(f"Yeni Resimler: {OUTPUT_IMG_DIR}")
print(f"Yeni Etiketler: {OUTPUT_LBL_DIR}")
