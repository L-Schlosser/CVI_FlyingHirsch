import os
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch

DATASET_PATH = Path(__file__).parent / ".." / ".." / "datasets" / "raw" / "thermal_data"
MODEL_SIZE = 'yolo11l.pt'
EPOCHS = 100
BATCH_SIZE = 4
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

def train_yolo_model(dataset_path, model_size='yolo11l.pt', epochs=100, batch_size=16, img_size=640):    
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
        'patience': 30,  # early stopping patience
        'save_period': 10,  # save model every 10 epochs
        'workers': 8,  # number of dataloader workers
        'project': 'runs/detect',  # project directory
        'name': 'ir_animal_detection',  # experiment name
        'exist_ok': True,  # overwrite existing experiment
        'pretrained': True,  # use pretrained weights
        'optimizer': 'AdamW',  # optimizer
        'lr0': 0.001,  # initial learning rate
        'lrf': 0.1,  # final learning rate factor
        'momentum': 0.937,  # momentum
        'weight_decay': 0.0005,  # weight decay
        'warmup_epochs': 3,  # warmup epochs
        'warmup_momentum': 0.8,  # warmup momentum
        'box': 3.0,  # box loss gain
        'cls': 0.2,  # class loss gain
        'dfl': 1.5,  # DFL loss gain
        'augment': True,  # apply augmentations

        # 'mosaic': 1.0,  # mosaic probability
        # 'mixup': 0.0,  # mixup probability                  was 0.1
        # 'copy_paste': 0.0,  # copy paste probability        was 0.1

        # --- FIXED THERMAL REGULARIZATION ---
        'mosaic': 0.8,           # Bring back partially to assist with small object scale
        'close_mosaic': 15,      # CRUCIAL: Disables mosaic for the final 10 epochs to clean up edge noise
        'mixup': 0.15,            # Keep to assist with small object scale and occlusion
        'copy_paste': 0.15,        # Keep to assist with small object scale and occlusion
        
        # --- SPATIAL VARIATION (Fights Overfitting) ---
        'scale': 0.6,            # Zooms in/out randomly to handle distance variation
        'translate': 0.1,        # Shifts images to handle framing variation
        'degrees': 15.0,         # Rotates images
        'fliplr': 0.5,           
        
        # --- COLOR/VAL VARIATION ---
        'hsv_h': 0.0,            # Keep off
        'hsv_s': 0.0,            # Keep off
        'hsv_v': 0.5,            # High value to simulate varying sensor gains/thermal contrast

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
        print(f"Best model saved at: runs/detect/ir_animal_detection/weights/best.pt")
        print(f"Validation mAP@0.5: {metrics.box.map50}")
        print(f"Validation mAP@0.5:0.95: {metrics.box.map}")
    except Exception as e:
        print(f"Error during training: {str(e)}")