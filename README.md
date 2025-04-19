# Medical Imaging RSNA Beast Cancer Detection
 Using Kaggle's RSNA Breast Cancer Dataset to detect breast cancer

🧠 Breast Cancer Detection Using Deep Learning
📁 Dataset
Kaggle’s Breast Cancer Screening dataset
Includes mammography images in .dcm (DICOM) and .jp2 (JPEG2000) formats.

🛠️ Project Pipeline
Data Conversion

Converted .dcm files to .png format for easier preprocessing.

Later experiments suggest .jp2 (JPEG2000) preserves more detail and may yield better model performance.

ROI Extraction

Used a pre-trained breast detection model (available on Kaggle) to extract Regions of Interest (ROIs) from the mammograms.

Exploratory Data Analysis

Conducted EDA to understand image distribution, label balance, and class imbalance.

Model Training

Trained three architectures:

ResNet50

EfficientNetB0

Vision Transformer (ViT)

Best overall performance: EfficientNetB0 (speed vs. accuracy trade-off).

Highest accuracy: Vision Transformer, but training time was prohibitive without high-end GPU (e.g., NVIDIA RTX 3090 or better).

📊 Evaluation Metric
ROC AUC Score was used to evaluate model performance, measuring the ability to distinguish between cancerous and non-cancerous images.

🚧 Future Work
💡 Data Augmentation with GANs: To address class imbalance by generating synthetic samples.

🤔 Diffusion Models: Considered, but converting .dcm to .png may cause loss of quality, especially when moving away from .jp2 benefits.

💬 Notes
Vision Transformer may outperform CNNs if you have the compute.

JPEG2000 format holds untapped potential in medical imaging, though support in deep learning pipelines is limited.

📌 To-Do
 Benchmark .jp2 directly vs .png

 Integrate GANs or VAEs for minority class synthesis

 Try training with CLIP embeddings or other multimodal setups
