import os
import cv2
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_dir', type=str, help='Folder contains original train dataset')
parser.add_argument('--test_dir', type=str, help='Folder contains original test dataset')
parser.add_argument('--train_save_dir', type=str, help='Folder save seq frames train dataset')
parser.add_argument('--test_save_dir', type=str, help='Folder save seq frames test dataset')
args = parser.parse_args()
# input_path = "/kaggle/dataset/public_test_data/public_test_gesture_data/data"


STRIDE = 1.0
MAX_IMAGE_SIZE = 640
N_FRAMES = 16
START_FRAME_SEC = 0.2
LABELS = ['VAR', 'look_at_me', 'reverse_signal', 'scratch', 'up_down', 'hand_fand', 'peekaboo', 'scissor', 'typing', 'wave_hand']

def get_frames_from_video(video_file):
    """
    video_file - path to file
    stride - i.e 1.0 - extract frame every second, 0.5 - extract every 0.5 seconds
    return: list of images, list of frame times in seconds
    """
#     root = input_path
#     video = cv2.VideoCapture(os.path.join(root, video_file))
    video = cv2.VideoCapture(video_file)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = frames/ fps
#     print(frames)
    i = duration * START_FRAME_SEC
    stride = (duration - i) / N_FRAMES
    images = []
    frame_times = []
    video.set(cv2.CAP_PROP_POS_MSEC, i * 1000)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            images.append(frame)
            frame_times.append(i)
            i += stride
            video.set(1, round(i * fps))
        else:
            video.release()
            break
    return images, frame_times


def resize_if_necessary(image, max_size=MAX_IMAGE_SIZE):
    """
    if any spatial shape of image is greater 
    than max_size, resize image such that max. spatial shape = max_size,
    otherwise return original image
    """
    if max_size is None:
        return image
    height, width = image.shape[:2]
    if max([height, width]) > max_size:
        ratio = float(max_size / max([height, width]))
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return image

# sample_video= '9.avi'
# images, frame_times = get_frames_from_video(sample_video)
# images = [resize_if_necessary(image, MAX_IMAGE_SIZE) for image in images]

def resize_if_necessary(image, max_size=MAX_IMAGE_SIZE):
    """
    if any spatial shape of image is greater 
    than max_size, resize image such that max. spatial shape = max_size,
    otherwise return original image
    """
    if max_size is None:
        return image
    height, width = image.shape[:2]
    if max([height, width]) > max_size:
        ratio = float(max_size / max([height, width]))
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return image

import warnings
warnings.filterwarnings('ignore')

# Extract frames from test dataset
def extract_test_data(root, save_path):
    for video_id in tqdm(os.listdir(root)):
        id = video_id.split('.')[0]
        sample_video = os.path.join(root, video_id)
        images, frame_times = get_frames_from_video(sample_video)
        os.makedirs(os.path.join(save_path, id), exist_ok=True)
        path2save = os.path.join(save_path, id)
        for i, (image, frame_time) in enumerate(zip(images, frame_times)):
            ps = os.path.join(path2save, f'{i}.jpg')
            image = resize_if_necessary(image)
            cv2.imwrite(ps, image)

# Extract frames from train dataset
def extract_train_data(root, save_path):
    for i, label in tqdm(enumerate(LABELS)):
        label_path = os.path.join(TRAIN_DIR, label)
        os.makedirs(os.path.join(save_path, str(i)), exist_ok=True)
        label_save_path = os.path.join(save_path, str(i))
        for j, video_id in enumerate(os.listdir(label_path)):
            id = video_id.split('.')[0]
            sample_video = os.path.join(label_path, video_id)
            images, frame_times = get_frames_from_video(sample_video)
            os.makedirs(os.path.join(label_save_path, f'{i}{j}'), exist_ok=True)
            path2save = os.path.join(label_save_path, f'{i}{j}')
            for k, (image, frame_time) in enumerate(zip(images, frame_times)):
                ps = os.path.join(path2save, f'{i}{j}{k}.jpg')
                image = resize_if_necessary(image)
                cv2.imwrite(ps, image)

if __name__ == '__main__':
    TRAIN_DIR = args.train_dir
    TEST_DIR = args.test_dir

    TEST_SAVE_PATH = args.test_save_dir
    TRAIN_SAVE_PATH = args.train_save_dir

    os.makedirs(TEST_SAVE_PATH, exist_ok=True)
    os.makedirs(TRAIN_SAVE_PATH, exist_ok=True)
    extract_train_data(TRAIN_DIR, TRAIN_SAVE_PATH)
    extract_test_data(root=TEST_DIR, save_path=TEST_SAVE_PATH)