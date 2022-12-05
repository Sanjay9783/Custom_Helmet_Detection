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
