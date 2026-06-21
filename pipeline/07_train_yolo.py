import os
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch

from config import BEST_WEIGHTS, MODEL_NAME, YOLO_RUNS_DIR, DATA_PATH_PROCESSED, ANNOTATED_IMAGES_SUBDIR

DATASET_PATH = DATA_PATH_PROCESSED / ANNOTATED_IMAGES_SUBDIR
MODEL_SIZE = 'yolo26s.pt'
EPOCHS = 70
BATCH_SIZE = 8
IMG_SIZE = 1024

def _verify_directory_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")
    
def _count_files_in_directory(path):
    print(f"Found {len(os.listdir(path))} files in {path}")

def create_dataset_config(dataset_path):
    dataset_path = os.path.abspath(dataset_path)
    
    train_images = os.path.join(dataset_path, 'images', 'train')
    val_images = os.path.join(dataset_path, 'images', 'val')
    train_labels = os.path.join(dataset_path, 'labels', 'train')
    val_labels = os.path.join(dataset_path, 'labels', 'val')

    _verify_directory_exists(train_images)
    _verify_directory_exists(val_images)
    _verify_directory_exists(train_labels)
    _verify_directory_exists(val_labels)
    
    _count_files_in_directory(train_images)
    _count_files_in_directory(val_images)
    _count_files_in_directory(train_labels)
    _count_files_in_directory(val_labels)
    
    config = {
        'train': train_images.replace('\\', '/'),
        'val': val_images.replace('\\', '/'),
        'nc': 1,
        'names': ['animal']
    }
    
    config_path = os.path.join(dataset_path, 'data.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path

def train_yolo_model(dataset_path, model_size='yolo11s.pt', epochs=100, batch_size=16, img_size=1024):    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and the correct drivers installed.")
        
    config_path = create_dataset_config(dataset_path)
    
    print(f"Loading model: {model_size}")
    model = YOLO(model_size)
    
    training_args = {
        'data': config_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': "cuda",
        'patience': 20,  # early stopping patience
        'save_period': 10,  # save model every 10 epochs
        'workers': 8,  # number of dataloader workers
        'project': str(YOLO_RUNS_DIR),  # project directory
        'name': MODEL_NAME,  # experiment name
        'exist_ok': True,  # overwrite existing experiment
        'pretrained': True,  # use pretrained weights
        'optimizer': 'MuSGD',
        'lr0': 0.00038,
        'lrf': 0.882,
        'momentum': 0.948,
        'weight_decay': 0.00027,
        'warmup_epochs': 0.98,

        'box': 9.83,
        'cls': 0.2,
        'dfl': 0.96,

        'augment': True,
        'mosaic': 0.992,
        'mixup': 0.05,
        'copy_paste': 0.404,
        'scale': 0.9,
        'fliplr': 0.304,
        'degrees': 0.0,
        'shear': 0.0,
        'translate': 0.275,

        'hsv_h': 0.0,
        'hsv_s': 0.0,
        'hsv_v': 0.2,

        'val': True,  # validate during training
        'plots': True,  # save training plots
        'verbose': True,  # verbose output

        'single_cls': True,  # treat all objects as a single class
    }
    
    print("Starting training...")
    print(f"Training parameters: {training_args}")
    
    results = model.train(**training_args)

    print("Evaluating model...")
    metrics = model.val()
    
    print("Exporting model...")
    model.export(format='onnx')
    
    return model, results, metrics

if __name__ == "__main__":
    try:
        model, results, metrics = train_yolo_model(
            dataset_path=DATASET_PATH,
            model_size=MODEL_SIZE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE
        )
        
        print("Training completed successfully!")
        print(f"Best model saved at: {BEST_WEIGHTS}")
        print(f"Validation mAP@0.5: {metrics.box.map50}")
        print(f"Validation mAP@0.5:0.95: {metrics.box.map}")
    except Exception as e:
        print(f"Error during training: {str(e)}")
