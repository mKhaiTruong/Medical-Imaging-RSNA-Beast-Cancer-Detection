# ============================================ Import ========================================
# 1. Handle datasets
import io
import os
import gc
import cv2
import time
import random
import pydicom
import dicomsdl
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
import tifffile as tiff
import imageio.v3 as iio
import SimpleITK as sitk
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing as mp
from collections import Counter
from joblib import Parallel, delayed
from pydicom.pixel_data_handlers.util import apply_voi_lut

# 2. Visualize datasets
import datetime as dtime
from datetime import datetime
import itertools
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
import plotly.figure_factory as pff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from IPython.display import display_html

# 3. Preprocess datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
## import iterative impute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
## fastai
# from fastai.data.all import *
# from fastai.vision.all import *

# 4. machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
## for classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import KBinsDiscretizer
from xgboost import XGBClassifier

# 5. Deep Learning
## Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import ViTModel, ViTFeatureExtractor, ViTForImageClassification

## Torch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet34, resnet50, ResNet50_Weights
from torchvision import datasets, transforms

# 6. metrics
import optuna
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.metrics import f1_score, r2_score
from sklearn.metrics import classification_report

# 7. ignore warnings and wandb 
import warnings
import wandb

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================================

# === Here lies some general functionalities ===
# === Seeds and Visualization ===
def set_seed(seed = 1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # On CuDNN we need 2 further options
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def show_values_on_bars(axs, h_v = 'v', space = 0.4):
    def _show_on_single_plot(ax):
        if h_v == 'v':
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                
                value = int(p.get_height())
                ax.text(_x, _y, format(value, ','), ha='center')
        elif h_v == 'h':
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_x() + p.get_height()
                
                value = int(p.get_width())
                ax.text(_x, _y, format(value, ','), ha='left')
    
    if isinstance(axs, np.ndarray):
        for i, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


# ============================================= WANDB ===========================================
def save_dataset_artifact(run_name, artifact_name, path, 
                          projectName = None, config = None, data_type = "dataset"):
    run = wandb.init(project=projectName,
                     name = run_name,
                     config= config)

    artifact = wandb.Artifact(name = artifact_name,
                              type = data_type)
    artifact.add_file(path)
    
    wandb.log_artifact(artifact)
    wandb.finish()
    print("Artifact has been save successfully!")

def create_wandb_plot(x_data = None, y_data = None, x_name = None, y_name = None,
                      title = None, log = None, plot = "line"):
    data = [
        [label, val] for (label, val) in zip(x_data, y_data)
    ]
    table = wandb.Table(data = data, columns = [x_name, y_name])
    
    if plot == "line":
        wandb.log({ log: wandb.plot.line(table, x_name, y_name, title=title) })
    elif plot == "bar":
        wandb.log({ log: wandb.plot.bar(table, x_name, y_name, title=title) })
    elif plot == "scatter":
        wandb.log({ log: wandb.plot.scatter(table, x_name, y_name, title=title) })

def create_wandb_hist(x_data = None, x_name = None, title = None, log = None):
    data = [[x] for x in x_data]
    table = wandb.Table(data = data, columns=[x_name])
    wandb.log({ log: wandb.plot.histogram(table, x_name, title=title) })
    
    
    
# ================================= AUGMENTATION =========================================
def transforms(isTrain = False):
    aug_list = []
    
    if isTrain:
        aug_list += [
            
            # === Spatial: Isotropic Scaling ===
            A.LongestMaxSize(max_size=224),
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
            
            # === Photometric Augmentations - Brightness / Contrast / Gamma — very mild ===
            A.OneOf([
                A.RandomToneCurve(scale=0.3, p=0.5),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.4, 0.5), p=0.5)
            ], p=0.5),
            
            # === Mild Contrast Enhancer (safe CLAHE) ===
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            
            # == Downscaling - Blurring (slight only) ==
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.Downscale(scale_min=0.9, scale_max=0.95, interpolation=dict(
                    upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_AREA), p=0.7),
            ], p=0.2),
            
            # === Occlusion-style Augmentation ===
            A.OneOf([
                A.GridDropout(ratio=0.2, unit_size_min=16, unit_size_max=32, random_offset=True, p=0.2),
                A.CoarseDropout(max_holes=6, max_height=0.15, max_width=0.25, min_holes=1, min_height=0.05, min_width=0.1, fill_value=0, mask_fill_value=None, p=0.25),
            ], p=0.3),
            
            # === Flips ===
            A.HorizontalFlip(p=0.5) if isTrain else A.NoOp(),
            A.VerticalFlip(p=0.5) if isTrain else A.NoOp(),
            
        ]
    
    else:
        # Inference-time: keep only resize + normalize
        aug_list += [
            A.LongestMaxSize(max_size=224),
            A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
        ]
    
    # === Normalize & ToTensor ===
    aug_list += [
        A.Normalize(),
        ToTensorV2()
    ]
    
    return A.Compose(aug_list)


# ==================================== RESNET50 ===========================================
class ResNet50Network(nn.Module):
    
    def __init__(self, outSize, no_columns):
        super().__init__()
        self.no_columns, self.outSize = no_columns, outSize
        
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # output: [B, 2048, 1, 1]
        # Remove the final classification layer to extract features (2048-dim)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # == metadata ==
        self.csv = nn.Sequential(
            nn.Linear(self.no_columns, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        
        # Classification
        self.classification = nn.Linear(2048 + 500, self.outSize)
    
    def forward(self, image, meta, prints=False):
        if prints: 
            print(f"Input Image Shape: {image.shape} \n Input Metadata Shape: {meta.shape}")
        
        # == Image CNN ==
        image = self.features(image)
        image = image.view(image.size(0), -1)   # Flatten → [B, 2048]
        if prints: print(f'Features image shape: {image.shape}')
        
        # == CSV FNN ==
        meta = self.csv(meta)
        if prints: print(f'Metadata shape: {meta.shape}')
        
        # Concatenate layers from image with layers from csv_data
        image_meta_data = torch.cat((image, meta), dim=1)
        if prints: print(f'Concatenated data: {image_meta_data.shape}')
        
        # == CLASSIF ==
        out = self.classification(image_meta_data)
        if prints: print(f'Out shape: {out.shape}')
        
        return out
    
# ======================================= EFFNET ===============================================
class EffNetNetwork(nn.Module):
    
    def __init__(self, outputSize, no_columns):
        super().__init__()
        self.no_columns, self.outputSize = no_columns, outputSize
        
        self.features = EfficientNet.from_pretrained('efficientnet-b3')
        
        self.csv = nn.Sequential(
            nn.Linear(self.no_columns, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(250, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        
        self.classification = nn.Sequential(nn.Linear(1536 + 250, self.outputSize))
    
    def forward(self, image, meta, prints=False):
        if prints: 
            print(f"Input Image Shape: {image.shape} \n Input Metadata Shape: {meta.shape}")
        
        # == Image CNN ==
        image = self.features.extract_features(image)
        image = F.avg_pool2d(image, image.size()[2:]).reshape(-1, 1536)
        if prints: print(f'Features image shape: {image.shape}')
        
        # == CSV FNN ==
        meta = self.csv(meta)
        if prints: print(f'Metadata shape: {meta.shape}')
        
        # Concatenate layers from image with layers from csv_data
        image_meta_data = torch.cat((image, meta), dim=1)
        if prints: print(f'Concatenated data: {image_meta_data.shape}')
        
        # == CLASSIF ==
        out = self.classification(image_meta_data)
        if prints: print(f'Out shape: {out.shape}')
        
        return out

# ===================================== Vision Transformer ==========================================
class ViTNetwork(nn.Module):
    
    def __init__(self, outSize, no_columns, model_name='google/vit-base-patch16-224-in21k'):
        super().__init__()
        self.no_columns = no_columns
        self.outSize = outSize
        
        # === ViT Backbone ===
        self.vit = ViTModel.from_pretrained(model_name)
        self.vit_hidden_size = self.vit.config.hidden_size
        
        # === CSV Metadata ===
        self.csv = nn.Sequential(
            nn.Linear(self.no_columns, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            
            nn.Linear(250, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        
        # === Final Classification Head ===
        self.classification = nn.Sequential(
            nn.Linear(self.vit_hidden_size + 250, self.outSize)
        )

    def forward(self, image, meta, prints=False):
        if prints: 
            print(f"Input Image Shape: {image.shape} \n Input Metadata Shape: {meta.shape}")

        # ViT expects images in [B, C, H, W]
        vit_out = self.vit(pixel_values=image).pooler_output  # shape: (B, vit_hidden_size)
        if prints: print(f'ViT output shape: {vit_out.shape}')
        
        meta_out = self.csv(meta)
        if prints: print(f'Metadata output shape: {meta_out.shape}')
        
        concat = torch.cat((vit_out, meta_out), dim=1)
        if prints: print(f'Concatenated shape: {concat.shape}')
        
        out = self.classification(concat)
        if prints: print(f'Final output shape: {out.shape}')
        
        return out
