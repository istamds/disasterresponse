## Table of Contents
* [Project Motivation](#project-motivation)
* [Installation](#installation)
* [File Descriptions](#file-descriptions)
* [Licensing, Authors and Acknowledgements](#licensing,-authors-and-acknowledgements)

## Project Motivation
When a disaster takes place messages are sent around. It is important to analyse those messages and inform the relief mechanisms in order for them to be able to act on time. This repository contains the code for the analysis of such messages. Using NLP and ML techniques the messages are classified in different categories, so that they can be forwarded fast to the right authority in respect. The whole approach is provided as a web application, where the user can give a message and receive the categories in which this message belongs.  

## Installation
In order to be able to run the code, the following python libraries are required:
* pandas
* sqlalchemy 
* nltk
* pickle
* numpy
* sklearn
* plotly
* json
* flask

## File Descriptions

1. ETL Pipeline: 
In the script process_data.py, a pipeline is programmed that loads the messages and categories datasets, merges them and cleans the data and at the end stores it in a SQLite database.
2. ML Pipeline: 
In the script train_classifier.py, you can find a machine learning pipeline that loads data from the SQLite database, builds a text processing and machine learning pipeline, trains and tunes a model and exports the final model as a pickle file.
3. Flask Web App: 
You are provided also with a flask web app. The app displays visualizations that describe the training data and uses the trained model to input text and return classification results.


### Licensing, Authors and Acknowledgements
The code provided, is free to be used as needed!
The datasets are provided by Figure Eight. More information can be found [here](https://appen.com/open-source-datasets/).