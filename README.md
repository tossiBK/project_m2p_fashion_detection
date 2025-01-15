# Project: DLBDSMTP01 – Project: From Model to Production

**Task 2: Image classification for a refund department (spotlight: Batch processing)**


## Requirements

Project was build with python 3.10. and the following packages:

* Jupyter Notebooks
* Pandas
* Numpy
* Scikit-Learn
* PyTorch (inclusive TorchVision & TorchAudio)
* MLFlow
* Pillow
* Pickle
* Flask

It also required:

* Docker (for the container build and API running)
* NVIDIA GPU (with CUDA support, if not, some changes need to be done in the whole project)

## Dataset source
Dataset used for the training: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
It is not included in the project, as by the size of the data.

Please copy the content of this into the folder:

```
data/fashion-dataset/
```

## Usage of the project

Follow the steps in the provided Jupyter Notebook. It will lead you through the whole process of the training and deployment of the API.

Install the script `batch_processing.py` into a server or container of your choice.
To set it up as a batch process every night use the following crontab on the server. Adjust the folder name and don´t forget to prepare the folders needed and access right for the script file.

```
5 0 * * * /usr/bin/python3 /opt/scripts/batch_processing.py
```

