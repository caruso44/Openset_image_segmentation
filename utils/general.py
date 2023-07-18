import torch

PATCH_SIZE = 64
PATCH_OVERLAP = 0.8
BATCH_SIZE = 16

PREFIX_LABEL = 'label'
PREFIX_OPT = 'opt'
PREFIX_LIDAR = 'lidar'

PREPARED_PATH = 'C:/Users/jpcar/OneDrive/Documentos/Área de Trabalho/IME/Pibt/Codigo/OpenMax-main/prepared'
PATCHES_PATH = "C:/Users/jpcar/OneDrive/Documentos/Área de Trabalho/IME/Pibt/Codigo/OpenMax-main/prepared/train_patches.npy"
PATCHES_VAL_PATH = "C:/Users/jpcar/OneDrive/Documentos/Área de Trabalho/IME/Pibt/Codigo/OpenMax-main/prepared/val_patches.npy"
PATCHES_TEST_PATH = "C:/Users/jpcar/OneDrive/Documentos/Área de Trabalho/IME/Pibt/Codigo/OpenMax-main/prepared/test_patches.npy"
MODEL_PATH = "C:/Users/jpcar/OneDrive/Documentos/Área de Trabalho/IME/Pibt/Codigo/OpenMax-main/open_set_model.pth"
DISCARDED_CLASS = 7
REMOVED_CLASSES = [10, 11]
N_CLASSES = 10 - len(REMOVED_CLASSES)
N_OPTICAL_BANDS = 4
VALID_SIZE = 0.15


#N_LIDAR_BANDS = 1#6
"""Bands order: 'BLUE','RED','GREEN','NIR','nx','ny','nz','curvatura','intensity','chm'"""
#                 0      1      2      3     0    1    2       3           4        5                     
MAX_EPOCHS = 500
EPOCHS = 70


#LEARNING_RATE = 1e-4
LEARNING_RATE = 4e-5
LEARNING_RATE_BETAS = (0.9, 0.999)
LEARNING_RATE_SCHEDULER_GAMMA = 0.995
#LEARNING_RATE_SCHEDULER_MILESTONES = [5, 10, 20, 50]
LEARNING_RATE_SCHEDULER_MILESTONES = [5, 20]

B_W = 0.01
T_W = 0.13
CLASSES_WEIGHTS = [B_W, T_W, T_W, T_W, T_W, T_W, T_W, T_W, T_W, 0]


EARLY_STOP_MIN_EPOCHS = LEARNING_RATE_SCHEDULER_MILESTONES[-1]
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 0.00009

PREDICTION_OVERLAPS = [0, 0.1, 0.3, 0.5, 0.7]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEN_VECTOR = 200000

IMAGE_SIZE = 64

NUM_KNOWN_CLASSES = 7