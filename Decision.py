import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def DecisionTreeModel(df, graphTitle):
    print("Decision Tree Classification")

    features = ["HighBP", "HighChol", "BMI", "Age", "GenHlth"]
    target = df.columns[0]
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
    
    model = DecisionTreeClassifier(random_state=19)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\nAnalysis Results for {graphTitle} Dataset:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for Decision Tree Classifier on {graphTitle} Dataset')
    plt.show()



df = pd.read_csv('Data/diabetes_012_health_indicators_BRFSS2015.csv')
DecisionTreeModel(df, "012_health_indicators")

df = pd.read_csv('Data/diabetes_binary_health_indicators_BRFSS2015.csv')
DecisionTreeModel(df, "Binary")

df = pd.read_csv('Data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
DecisionTreeModel(df, "Binary 50/50 Split")
