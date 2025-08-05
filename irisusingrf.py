import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import mlflow
import matplotlib.pyplot as plt
import joblib

import dagshub
dagshub.init(repo_owner='ravishyadav112', repo_name='mlflow', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
# mlflow.set_tracking_uri("https://dagshub.com/ravishyadav112/mlflow.mlflow")

data = load_iris()

df = pd.DataFrame(data.data , columns=data.feature_names)
X = df
y = data.target

X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.3 , random_state=2)





max_depth = 5
n_estimator = 50
mlflow.set_experiment("RandomForest")


with mlflow.start_run():

    classifier = RandomForestClassifier(n_estimators=n_estimator, max_depth= max_depth)
    classifier.fit(X_train , y_train)
    y_pred = classifier.predict(X_test)

    plt.figure(figsize=(8,8))
    cm  = confusion_matrix(y_pred , y_test)
    sns.heatmap(cm , annot=True , fmt='d' , cmap='coolwarm' , xticklabels=data.target_names , yticklabels=data.target_names ) , 
    plt.xlabel('Predicted')
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusionmatrix.png")
    mlflow.log_artifact("confusionmatrix.png")
    acc = accuracy_score(y_pred , y_test)
    
    mlflow.log_metrics({'Accuracy' : acc})

    joblib.dump(classifier , "RandomForest.pkl")
    mlflow.log_artifact("RandomForest.pkl")

    mlflow.set_tags({'author' : "Ravish" , "model" : "Random Forest"})
    
    

    

    mlflow.log_params({"n_estimators":n_estimator , "Max Depth" : max_depth })
    print("confusion_matrix\n" , cm)
