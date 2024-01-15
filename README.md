## Project Title
UNSW-NB15 Network Intrusion Detection System

## Description
This project involves the use of PySpark, a Python library for Apache Spark, to perform machine learning tasks. It uses various machine learning algorithms for classification tasks.

## Machine Learning Algorithms Used
1. Logistic Regression
2. Support Vector Machine
3. Random Forest Classifier
4. Artificial Neural Network

## Objective:
The objective of this project is to build a Network Intrusion Detection System (NIDS) using machine learning algorithms. The NIDS is trained on the UNSW-NB15 dataset, which contains network traffic data.
The NIDS is trained to classify network traffic as either normal or malicious.

## Steps Taken

1. **Environment Setup**: The Python executable path is set for both PySpark's Python and driver Python.

2. **Library Imports**: Essential libraries for data manipulation, visualization, and machine learning are imported. These include pandas, numpy, seaborn, matplotlib, and various components from PySpark MLlib.

3. **Data Processing**: The project uses PySpark's DataFrame API for data processing. The data is read from a CSV file and converted to a PySpark DataFrame. The DataFrame is then cleaned and prepared for machine learning tasks.

4. **Feature Engineering**: The project uses StandardScaler, VectorAssembler, and StringIndexer for feature transformation.

5. **Model Building**: Different machine learning models like RandomForestClassifier, LogisticRegression, LinearSVC, and MultilayerPerceptronClassifier are used.

6. **Model Evaluation**: The models' performances are evaluated using MulticlassClassificationEvaluator, BinaryClassificationEvaluator, and BinaryClassificationMetrics.

7. **Hyperparameter Tuning**: ParamGridBuilder and CrossValidator are used for hyperparameter tuning.


## Results

The project involved training and tuning several machine learning models, and their performance was evaluated using various metrics. 

1. **Default Model(LR) vs Best-Tuned Model(LR) vs Weighted Model(LR)**: The Best-Tuned Model outperformed the Default and Weighted Models in all metrics. It achieved an AUC of 0.935752, sensitivity of 0.877980, and accuracy of 0.866021. The improvement in performance after tuning indicates that the tuning process was successful in optimizing the model's parameters.

2. **SVM Model vs Best SVM Model**: The performance of the SVM Model and the Best SVM Model was identical across all metrics. Both models achieved an AUC of approximately 0.9287, sensitivity of 0.884047, and accuracy of 0.845775. This suggests that the tuning process did not lead to any improvements for the SVM model.

3. **Default Model(RF) vs Fine-Tuned Model(RF)**: The Fine-Tuned Model outperformed the Default Model in most metrics, with a higher sensitivity (0.933116 vs 0.903420), precision (0.954716 vs 0.969437), F1-Score (0.943792 vs 0.935265), and accuracy (0.924353 vs 0.914880). However, the Default Model had a slightly higher AUC and AreaUnderPR.

4. **ANN Model**: The ANN Model achieved an AUC of 0.957164, sensitivity of 0.954835, and accuracy of 0.918781. These results suggest that the ANN Model performed well on the task.

In summary, the Best-Tuned Model(LR) and the Fine-Tuned Model(RF) achieved the highest performance among all models, with the Best-Tuned Model(LR) achieving the highest AUC and the Fine-Tuned Model(RF) achieving the highest accuracy.