import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

def nBayes(df):

    features = ["HighBP", "HighChol", "BMI", "Age", "GenHlth"]
    target = df.columns[0]
    X = df[features]
    Y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=60)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nby_pred = nb_model.predict(X_test)

    print("Naive Bayes Accuracy: ", accuracy_score(y_test, nby_pred))
    print("Naive Bayes F1 Score: ", f1_score(y_test, nby_pred, average='weighted'))
    print("Naive Bayes Recall Score: ", recall_score(y_test, nby_pred, average='weighted'))
    print("Naive Bayes Precision Score: ", precision_score(y_test, nby_pred, average='weighted'))

    print(classification_report(y_test, nby_pred))

    cm = confusion_matrix(y_test, nby_pred)
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title('Naive Bayes Confusion Matrix')
    plt.show()


df = pd.read_csv('Data/diabetes_012_health_indicators_BRFSS2015.csv')
nBayes(df)
df2 = pd.read_csv('Data/diabetes_binary_health_indicators_BRFSS2015.csv')
nBayes(df2)
df3 = pd.read_csv('Data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
nBayes(df3)