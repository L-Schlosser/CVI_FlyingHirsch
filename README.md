# ALFS YOLO Detection Pipeline

## Setup
pip install -r requirements.txt

## check dataset
python check_dataset.py

## convert labels - do ONLY once 
python convert_labels.py

## replace labels folder
dataset/labels/  →  dataset/labels_converted/

## check dataset again
python check_dataset.py

## Train
python train.py

## Predict
python predict.py

## Dataset format
YOLO format:
<class_id> <x_center> <y_center> <width> <height>