# superai-3-DataSci-Bigdata

## Introduction

  The Super AI Engineer 2022: Data Science and Big Data Challenge is a competition for participants to develop strategies for predicting car ownership from a questionnaire dataset. In this notebook, I used AutoGluon, an open-source library developed by Amazon Web Services, to optimize hyperparameters and ensemble methods for our machine learning model. AutoGluon simplifies the process of building and deploying machine learning models by providing a high-level interface for training deep learning models using various frameworks and tools for automating hyperparameter optimization and ensemble learning. The competition dataset is not publicly available, so participants must provide their own dataset. This notebook is intended to be a resource and inspiration for other participants in the competition.
  
 ## Outcome

- **Ranked 🎖️#1 amongs 208 teams** participating the [Kaggle hackathon](https://www.kaggle.com/competitions/hackathon-online-data-science-and-big-data/leaderboard) in the score leaderboard.
- **Scored 0.80076** in both private and public leaderboard

![image](https://user-images.githubusercontent.com/98932144/208222890-11a07548-74ab-41ce-b369-58bf83d023cf.png)

## Requirements
- AutoGluon 

##### You can install the following code:

``` python
!pip -q install autogluon 
```
AutoGluon is an open-source library developed by Amazon Web Services (AWS) that aims to make it easier for developers to build and deploy machine learning models. It provides a high-level interface for training and deploying deep learning models using a variety of different frameworks, including PyTorch, MXNet, and Gluon.

<img src="https://user-images.githubusercontent.com/98932144/208224030-41b6c703-0db2-44b5-87a8-59ef29686ec8.png" width="300" height="75">

One of the key features of AutoGluon is its ability to automate the process of hyperparameter optimization. This means that it can search for the best values for parameters such as learning rate, batch size, and number of epochs, without the need for manual tuning. This can save a lot of time and effort, especially for complex models or large datasets.


AutoGluon provides a TabularPrediction class that can be used to train and deploy machine learning models on tabular datasets, which are datasets that consist of rows and columns of data. Here is an example of how to use AutoGluon's TabularPrediction class to train a model on a tabular dataset:

``` python
import autogluon as ag
from autogluon import TabularPrediction as task

# Load the data
dataset = task.Dataset(file_path='path/to/data.csv')

# Split the data into training and validation sets
train_data, test_data = dataset.split(0.8)

# Create the model
model = task.fit(train_data=train_data, label='target_column', eval_data=test_data)

# Make predictions on the test set
predictions = model.predict(test_data)

# Evaluate the model
accuracy = model.evaluate(test_data)
print(f'Test accuracy: {accuracy:.4f}')
``` 
#### For best accuracy mode you can adjust this :

``` python
# Create the model
model = task.fit(train_data=train_data, label='target_column', eval_data=test_data,
                 auto_stack=True, hyperparameter_tune=True)
```

function to train the model in "best accuracy" mode. The auto_stack and hyperparameter_tune arguments enable automated ensemble learning and hyperparameter optimization, respectively. The label argument specifies the name of the target column in the dataset, which is the column that we want to predict. The predict function will make predictions on the test set, and the evaluate function will compute the model's accuracy on the test set.

##### Note that the "best accuracy" mode may take longer to train the model, as it will perform a more thorough search for the optimal hyperparameters and ensemble methods. However, this can often lead to better results and higher accuracy.


