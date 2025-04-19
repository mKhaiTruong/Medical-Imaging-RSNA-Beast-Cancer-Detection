# 🧠 Breast Cancer Detection Using Deep Learning

## 📁 Dataset
Kaggle’s Breast Cancer Screening dataset  
Includes mammography images in `.dcm` (DICOM) and `.jp2` (JPEG2000) formats.

---

## 🛠️ Project Pipeline

1. **Data Conversion**
   - Converted `.dcm` files to `.png` format for easier preprocessing.
   - Later experiments suggest `.jp2` (JPEG2000) preserves more detail and may yield better model performance.

2. **ROI Extraction**
   - Used a pre-trained breast detection model (available on Kaggle) to extract Regions of Interest (ROIs) from the mammograms.

3. **Exploratory Data Analysis**
   - Conducted EDA to understand image distribution, label balance, and class imbalance.

4. **Model Training**
   - Metrics used: Accuracy, Loss and ROC score
   - ROC score (Receiver Operating Characteristic score) in medical imaging measures the diagnostic ability of a model by plotting the true positive rate against the false positive rate at various thresholds. It helps evaluate how well the model distinguishes between diseased and healthy cases. In terms of medical images, where false postive cases can lead to a bad deal, ROC score might be more crucial than Loss.
   - Trained three architectures:
     - **ResNet50**
     - **EfficientNetB0**
     - **Vision Transformer (ViT)**
   - **Best overall performance:** EfficientNetB0 (speed vs. accuracy trade-off).
   - **Highest ROC score:** Vision Transformer, but training time was prohibitive without high-end GPU (e.g., **NVIDIA RTX 4000x or better**).

---

## 📊 Evaluation Metric

- **ROC AUC Score** was used to evaluate model performance, measuring the ability to distinguish between cancerous and non-cancerous images.

---

## 🚧 Future Work

- 💡 **Data Augmentation with GANs:** To address class imbalance by generating synthetic samples.
- 🤔 **Diffusion Models:** Considered, but converting `.dcm` to `.png` may cause loss of quality, especially when moving away from `.jp2` benefits.

---

## 💬 Notes

- Vision Transformer may outperform CNNs **if you have the compute**.
- JPEG2000 format holds **untapped potential** in medical imaging, though support in deep learning pipelines is limited.

---

## 📌 To-Do

- [ ] Benchmark `.jp2` directly vs `.png`
- [ ] Integrate GANs or VAEs for minority class synthesis
- [ ] Try training with CLIP embeddings or other multimodal setups
