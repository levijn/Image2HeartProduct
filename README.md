# Image to Heart 


## Introduction
The goal of this project is to automatically segment the heart of a patient. This is done by training a fully convolutional network on labeled MRI scans of the heart.  


## Setting up project
**Creating a virtual environment**

To use this code you first have to make a virtual environment in the same folder you placed this repository. To create a virtual environment you can look at some tutorials or the python documentation.

**Downloading packages**

In the 'Image2HeartProduct' folder you will find a requirements.txt file. These are the packages required to run the code except for Pytorch. To download PyTorch you can check there <a href="https://pytorch.org/get-started/locally/">Website</a>. To automatically download the packages you can run the command 

```
pip install -r requirements.txt
```

**Pretrained weights**
The weights file is not included in this repository due to its size. To get the weights you can either train it yourselfs or send an email to request them. 

## Running the model

**Training the model**

After setting up the project you can train the model using train_model_finetuning.py. The current parameters work the best, but if you  want to edit them you can find them in the main function. This will automatically save the model when it has finished training. 


**Using model on patients**

After training you can used the saved model to segment the cardiac structures of patients. This can be done by running the run_model.py file. Make sure to check if the name of the model is correct.
