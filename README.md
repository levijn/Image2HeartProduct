# Image to Heart 


## Introduction
The goal of this project is to automatically segment the heart of a patient. This is done by training a   fully convolutional network on labeled MRI scans of the heart.  


<br/>

## Setting up the project

### Virtual environment

**Creating a virtual environment**

To use this code you first have to make a virtual environment in the folder named 'Image2HeartProduct'. This can be done by following these steps:

1. Open a new Command Prompt(cmd) and make sure that the filepath you are running in had the same format as the following: 'C:\\.....\Image2HeartProduct>'.
2. Run the following command:
   ```
   py -m venv venv
   ```
This will create a new folder in the explorer named 'venv'. Do not change this name.

<br/>

**Activating the virtual environment** 

After the creation of the virtual environment 'venv' it needs to get activated to work in it. This can be done by doing the following:

In your Command Prompt enter the command:
```
venv\Scripts\activate.bat
```
Your Command Prompt should now be in the format:  '(venv) C:\\.....\Image2HeartProduct>'

<br/>

**Downloading the packages**

In the 'Image2HeartProduct' folder you will find a 'requirements.txt' file. These are the packages required to run the code. To automatically download the packages first make sure you are still in the Command Prompt that is still in the format:  '(venv) C:\\.....\Image2HeartProduct>'

Then run the following command:

```
pip install -r requirements.txt
```

*For further information about creating a virtual environment you can look at some tutorials or the python documentation.*

<br/>

***Pretrained weights***

The weights file is not included in this repository due to its size. To get the weights you can either train it yourself or send an email to request them. 

<br/>

## Running the model

**Training the model**

After setting up the project you can train the model using 'train_model.py' within the folder 'Code/Model'. The current parameters work the best, but edditing these can be done inside the main function. After training the trained weights will automatically be saved in a separate '.H5' file.

<br/>

**Running the model**

After training you can used the saved model to segment the cardiac structures of patients. This can be done by running the 'run_model.py' file. Make sure to check if the name of the weights corresponds to the file in which the weights have been trained after training.

<br/>

**Evaluating the model**

The file 'evaluate_model.py' can be used to evaluate how accurate the trained model is. This file can be used  to evaluate the Dice scores while training and to calculate the Dice score of your pretrained model.
