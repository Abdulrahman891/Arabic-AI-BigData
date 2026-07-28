# Detection of AI-Generated Arabic Text Using Machine Learning

A machine learning and Big Data project focused on detecting whether Arabic text is human-written or AI-generated.
The project combines Arabic text preprocessing feature engineering machine learning and distributed processing using Apache Spark. It also includes a real-time detection pipeline using Spark Structured Streaming.

## About the Project

The project was developed to explore scalable approaches for detecting AI-generated Arabic text.
Arabic text is cleaned and preprocessed before extracting TF-IDF and stylometric features. Multiple machine learning models are then trained and compared using Spark MLlib to identify the best performing approach.
Docker was used to create a consistent local environment for running the Apache Spark pipeline and supporting the Big Data processing workflow.
The final system also supports real-time classification of incoming Arabic text through a streaming pipeline.

## Dataset

The project uses 51,743 Arabic text samples containing both human-written and AI-generated content.
- Training: 36,220 samples
- Validation: 7,761 samples
- Testing: 7,762 samples

## System Architecture
The project follows an end-to-end distributed pipeline:
Data → Preprocessing → Feature Engineering → Machine Learning → Prediction → Real-Time Streaming → AI Text Detection

<img width="337" height="103" alt="image" src="https://github.com/user-attachments/assets/a94ab5f7-25da-4193-8559-81511f10153f" />

## Feature Engineering

The project combines textual and writing-style features to improve classification.
- TF-IDF text representation
- Average word length
- Punctuation usage
- Question marks
- Grammatical person characteristics
- Entity diversity

<img width="344" height="95" alt="image" src="https://github.com/user-attachments/assets/69b287a1-8d54-4d14-acd4-25597c8669cd" />

## Machine Learning Models

Three machine learning models were trained and evaluated using Spark MLlib:
- Logistic Regression
- Random Forest
- Linear SVM

## Results

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 97.62% | 97.65% |
| Random Forest | 83.00% | 77.45% |
| Linear SVM | 98.19% | 98.20% |

Linear SVM achieved the best overall performance with 98.19% accuracy and 98.20% F1 score.

<img width="540" height="450" alt="image" src="https://github.com/user-attachments/assets/7d0c6739-cf7a-46cb-8ee8-3b8afe2c73a6" />

## Development Environment

Docker was used to set up the local Big Data environment and provide a consistent setup for running Apache Spark and the project pipeline.

<img width="1233" height="190" alt="image" src="https://github.com/user-attachments/assets/39325799-37ed-43f5-be73-d4c4dca10752" />

## Real-Time Detection

A real-time prediction pipeline was implemented using Spark Structured Streaming.
Incoming Arabic text is automatically preprocessed and transformed before the trained model classifies it as human-written or AI-generated.

<img width="331" height="103" alt="image" src="https://github.com/user-attachments/assets/59a3c804-effb-45c0-ba25-244ca7036ffb" />

## Tools Used

- Python
- Apache Spark
- Spark MLlib
- Spark Structured Streaming
- Docker
- NLP
- TF-IDF
- Parquet
