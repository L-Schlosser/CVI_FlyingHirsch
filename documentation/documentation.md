1. **Get initial dataset:**

wget -O dataset.zip "https://zenodo.org/records/18772136/files/dataset.zip?download=1"

1. **Unzip the dataset into `datasets/raw`:**

unzip dataset.zip -d datasets/raw

The dataset is now available in the `datasets` directory. It contains the following files:

- /images/test train val 
- /labels/test train val

1. **Check the dataset with `01_check_dataset.py`** 

1. **Convert labels - 02.convert_labels.py**

1. **Explore ALFS dataset with `03_exploration_ALFS.ipynb`**

1. **Get new dataset:**

wget -O dataset.zip "https://zenodo.org/records/19034999/files/dataset.zip?download=1"

rename old datasets/raw to datasets/raw_old

unzip dataset.zip -d datasets/raw

1. **Initialize same Train/Test/val split with copy_data_oneClass.py**

-> to get the same initial Train/Test/val split from the AFLS dataset, we ran copy_data_oneClass.py for the new dataset

1. **Explore new dataset with 03_exploration.ipynb**

1. **Preprocess dataset with 04_preprocessing.ipynb**

1. **Explore preprocessed dataset with 04b_preprocessingExploration.ipynb**

1. **Train model with 05_train.py**

- We saved three different profiles in `config.py`that we used for training the model. The profiles are:
    - "fast"
    - "balanced"
    - "quality"

- For training the model, we had several approaches:
    - first we trained the model using initial COCO weights yolo26s/m/l.pt -> 

    ``` python
    model = YOLO("yolo26l.pt")
    model.train(**build_train_args())
    ```
    - then we trained the model using the best weights from the previous training -> 

    ``` python
    model = YOLO("runs/detect/train/yolo26l_annotated/weights/best.pt")
    model.train(**build_train_args())
    ```
    - also we used the approach to adapt the first layer from 3 channels to 1 channel (for thermal images) ->

    ``` python

        model = YOLO("yolo26l.pt")

    #try out:
    first = model.model.model[0]
    print(first.conv)

    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=first.conv.out_channels,
        kernel_size=first.conv.kernel_size,
        stride=first.conv.stride,
        padding=first.conv.padding,
        bias=first.conv.bias is not None
    ).to(first.conv.weight.device)

    print(new_conv)

    
    with torch.no_grad():
        new_conv.weight.copy_(
            first.conv.weight.mean(dim=1, keepdim=True)
        )

    model.model.model[0].conv = new_conv
    print(model.model.model[0].conv.in_channels)

    model.train(**build_train_args())

    ```

1. **Validate model with 06_validate.py**

1. **run prediction with 07_predict.py**

1. **run tracking with 08_tracking.py**

1. **visualize trackings**

    - visualize flight ID with 09_10_visualizeTracking.ipynb
    - visualize flight ID with 09_276_visualizeTracking.ipynb









