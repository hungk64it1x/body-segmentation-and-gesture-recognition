import os
import numpy as np

class CFG:
    TRAIN_DIR = './dataset/frame_dataset/train'
    TEST_DIR = './dataset/private_test/frame_dataset'
    MASK_TRAIN_DIR = './dataset/segment_dataset/train'
    MASK_TEST_DIR = './dataset/private_test/segment_dataset'
    NUM_CLASSES = 10
    IMAGE_SIZE = 256
    N_FOLDS = 5
    SEED = 42
    TARGET_COL = 'label'
    SEQ_LEN = 16
    BATCH_SIZE = 12
    NUM_EPOCHS = 12
    NUM_WORKERS = 4
    FRAMES = np.arange(0, SEQ_LEN, 1).tolist()
    LABELS = ['VAR', 'look_at_me', 'reverse_signal', 'scratch', 'up_down', 'hand_fand', 'peekaboo', 'scissor', 'typing', 'wave_hand']
    BACKBONE = 'dm_nfnet_f0'
    DROP_RATE = 0.1
    DROP_PATH_RATE = 0.0
    PRETRAINED = True
    LSTM_HIDDEN_SIZE = 128
    LSTM_LAYERS = 1
    LSTM_DROP = 0.0
    SCHEDULER = 'CosineAnnealingWarmRestarts'
    FACTOR = 0.2 
    PATIENCE = 4 
    WEIGHT_DECAY = 1e-6
    EPS = 1e-6 
    T_MAX = 12 
    T_0 = 10 
    BASE_LR = 1e-4
    MIN_LR = 1e-8
    WARM_UP = 3
    TTA = True
    NUM_TTA = 5

    DEVICE = 'cuda'
    OUTPUT_DIR = f'./checkpoint/{BACKBONE}'
    os.makedirs(OUTPUT_DIR, exist_ok=True)