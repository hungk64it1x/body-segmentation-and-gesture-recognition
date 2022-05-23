cd download
unzip gesture_training_data.zip
cp -r gesture_data/* ../cls/dataset/video_dataset
cd ../cls/video_dataset&&mkdir -p public_test_data/data
cp -r ../../download/public_test_gesture_data/data/* ./public_test_data/data/
cd ../../download

cd download
unzip train_segment.zip
mkdir -p ../segmentation/dataset/train/images
mkdir -p ../segmentation/dataset/train/masks 
mkdir -p ../segmentation/dataset/test/images
cp -r segment_data/image/* ../segmentation/dataset/train/images
cp -r segment_data/mask/* ../segmentation/dataset/train/masks 
unzip private_test_data.zip
cd ./private_test_data
unzip private_gesture_data.zip
unzip private_segment_data.zip
cp -r ./private_gesture_data/private_gesture_data/data/* ../../cls/dataset/private_test/video_dataset
cp -r ./private_segment_data/image/* ../../segmentation/dataset/private_test




