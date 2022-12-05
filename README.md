<br />
<div align="center">
  
  <h3 align="center">Custom_Helmet_Detection</h3>

  <p align="center">
    Computer vision Project
    <br />
    <a href="https://github.com/Sanjay9783/Custom_Helmet_Detection.git"><strong>Explore the Repo »</strong></a>
    <br />
    <a href="https://github.com/Sanjay9783/Animal_10_Classification/blob/main/model.ipynb"> Model Building</a>
  </p>
</div>

## Introduction

Training a model for custom object detection (TF 2.x) on Google Colab for automating the detection of whether people are wearing helmet or not, using kaggle custom helmet detection dataset.

## Dataset prepration

1. loading dataset: This dataset contains 764 images of 2 distinct classes for the objective of helmet detection.Bounding box annotations are provided in the PASCAL VOC format. [Link to kaggle dataset](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)
2. Once you’ve collected all the images needed, if dataset dosent contains bounding box we need to label them manually. There are many packages that serve this purpose. labelImg is a popular choice.
3. Create Label Map (.pbtxt): Classes need to be listed in the label map for each and ever class in dataset.
    ```shell
    item {
        id: 1
        name: 'With Helmet'
    }

    item {
        id: 2
        name: 'Without Helmet'
    }
   ```
4. Create TFRecord (.record): TFRecord is an important data format designed for Tensorflow. The dataset needs to be translated into the TFRecord format before training a custom object detector.

## Drive prepration

   ```shell
   from google.colab import drive
   drive.mount('/content/drive')
   
   # clone the tensorflow models on the colab cloud vm
   !git clone --q https://github.com/tensorflow/models.git
   
   # navigate to /models/research folder to compile protos
   %cd /content/models/research
   
   # Compile protos.
   !protoc object_detection/protos/*.proto --python_out=.
   
   # COCO API installation
   !git clone https://github.com/cocodataset/cocoapi.git
   
   %cd /content/models/research/cocoapi/PythonAPI
   !make
   !cp -r pycocotools /content/models/research
   
   %cd /content/models/research
   # Install TensorFlow Object Detection API.
   !cp object_detection/packages/tf2/setup.py .
   !python -m pip install .
   ```

## Download pre-trained model

The model zoo contains a large number of pre-trained object identification models. The models need to be restored in Tensorflow using their checkpoints (.ckpt files), which are records of prior model states, in order to train them using our custom data set.

For this project i have used `SSD MobileNet V2 FPNLite 320x320` pre-trained model for transfer learning on my custom data set.

## Training

Before training there need to be changes to be made in pipeline.config file.

```shell
num_classes: # Set this to the number of different label classes
batch_size: 8 # Increase/Decrease this value depending on the available memory (Higher values require more memory and vice-versa)
fine_tune_checkpoint: # Path to checkpoint of pre-trained model
fine_tune_checkpoint_type: "detection" # Set this to "detection" since we want to be training the full detection model
use_bfloat16: false # Set this to false if you are not training on a TPU

train_input_reader {
    label_map_path: "annotations/label_map.pbtxt" # Path to label map file
    tf_record_input_reader {
    input_path: "annotations/train.record" # Path to training TFRecord file
    }

eval_input_reader {
     label_map_path: "annotations/label_map.pbtxt" # Path to label map file
     shuffle: false
     num_epochs: 1
     tf_record_input_reader {
     input_path: "annotations/test.record" # Path to testing TFRecord
     }
```

training: python model_main_tf2.py --model_dir=[path of model directory] --pipeline_config_path=[path for pipeline_config]
