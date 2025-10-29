

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

def KNeighbor(df,graphTitle):
    print("K Nearest Neighbor Classification")
    features = ["HighBP", "HighChol", "BMI", "Age", "GenHlth", "PhysActivity", "Income", "DiffWalk"]
    target = df.columns[0]  
    x = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=19)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(x)
    scores = []
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(f"Analysis Results for {graphTitle} Dataset:")
    print(classification_report(y_test, y_pred))
    cm=confusion_matrix(y_test, y_pred)
    print(cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for KNN Classifier on {graphTitle} Dataset')
    plt.show()

#Find best K
'''
    for k in range(2, 21):
        print("k=", k)
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X_scaled, y, cv=5)
        scores.append(np.mean(score))
 
    sns.lineplot(x=range(2, 21), y=scores, marker='o') 
    plt.title(f'KNN Classifier Accuracy for {graphTitle} Dataset')
    plt.xlabel("K Values")
    plt.ylabel("Accuracy Score")
    plt.show()
'''


df = pd.read_csv('Data/diabetes_012_health_indicators_BRFSS2015.csv')
KNeighbor(df, "012_health _indicators")

df = pd.read_csv('Data/diabetes_binary_health_indicators_BRFSS2015.csv')
KNeighbor(df,"Binary")

df = pd.read_csv('Data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
KNeighbor(df,"Binary 50/50 split")