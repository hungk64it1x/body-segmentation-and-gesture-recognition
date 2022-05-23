import gdown
from tqdm import tqdm

url = 'https://drive.google.com/uc?export=download&id=17Q49B2bZxVHVq-Qwh2YKOlt-ERBBd4m1'
output = './download/train_gesture.zip'
gdown.download(url, output, quiet=False)

# url = 'https://drive.google.com/uc?export=download&id=10FtxHpO-qDJ5i3J0s8-I2uXmh6cPHlK_'
# output = './download/test.zip'
# gdown.download(url, output, quiet=False)

# url = 'https://drive.google.com/uc?export=download&id=1Wir7xL9t0jQNalbwHHjk5hrzXgrWno7M'
# output = './download/train_segment.zip'
# gdown.download(url, output, quiet=False)

# url = 'https://drive.google.com/uc?export=download&id=1sjeF1jB5se7bKfCxgeXrdkVlvOtd9SxL'
# output = './download/private_dataset.zip'
# gdown.download(url, output, quiet=False)
