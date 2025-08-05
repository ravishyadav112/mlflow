import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

data = load_iris()

df = pd.DataFrame(data.data , columns=data.feature_names)
X = df
y = data.target
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.3 , random_state=2)





max_depth = 5
criterion = 'log_loss'
splitter ='best'
mlflow.set_experiment("Decision_Tree")


with mlflow.start_run():

    classifier = DecisionTreeClassifier(criterion=criterion , max_depth= max_depth,splitter=splitter)
    classifier.fit(X_train , y_train)
    y_pred = classifier.predict(X_test)


    cm  = confusion_matrix(y_pred , y_test)
    acc = accuracy_score(y_pred , y_test)
    report = classification_report(y_pred , y_test)
    mlflow.log_metrics({'Accuracy' : acc})
    mlflow.sklearn.log_model(classifier , artifact_path="Classifier")
    
    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    mlflow.log_params({"Criterion" : criterion , "Max Depth" : max_depth , "Splitter" : splitter})
    print("confusion_matrix\n" , cm)