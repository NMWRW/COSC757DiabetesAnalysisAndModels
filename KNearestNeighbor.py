

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import train_test_split

def KNeighbor(df,graphTitle):
    print("K Nearest Neighbor Classification")
    df = pd.read_csv('Data/diabetes_012_health_indicators_BRFSS2015.csv')
    features = ["HighBP", "HighChol", "BMI", "Age", "GenHlth", "PhysActivity", "Income", "DiffWalk"]
    target = "Diabetes_012"
    x = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=19)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    K = []
    training = []
    test = []
    scores = {}

    for k in range(2, 21):
        print(f"K Value: {k}")
        clf = KNeighborsClassifier(n_neighbors = k)
        clf.fit(X_train, y_train)

        training_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        K.append(k)

        training.append(training_score)
        test.append(test_score)
        scores[k] = [training_score, test_score]
        
    for keys, values in scores.items():
        print(keys, ':', values)

    ax = sns.stripplot(x=K, y=training)  # Use x and y as keyword arguments
    ax.set(xlabel='Values of k', ylabel='Training Score')
    plt.show()
    plt.scatter(K, training, color='k')
    plt.scatter(K, test, color='g')
    plt.title(f'KNN Training and Test Scores for {graphTitle} Dataset')
    plt.show()
    
df = pd.read_csv('Data/diabetes_012_health_indicators_BRFSS2015.csv')
KNeighbor(df,"Full3Classes")
df = pd.read_csv('Data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
KNeighbor(df,"50/50")
df = pd.read_csv('Data/diabetes_binary_health_indicators_BRFSS2015.csv')
KNeighbor(df,"Binary")