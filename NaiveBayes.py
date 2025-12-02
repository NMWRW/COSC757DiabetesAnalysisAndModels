import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

df = pd.read_csv('Data/diabetes_012_health_indicators_BRFSS2015.csv')
df2 = pd.read_csv('Data/diabetes_binary_health_indicators_BRFSS2015.csv')
df3 = pd.read_csv('Data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

#First dataframe
X = df.drop(columns=['Diabetes_012'])
Y = df['Diabetes_012']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=60)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

nb_model = GaussianNB()
nb_model.fit(X_train_scaled,y_train)
nby_pred = nb_model.predict(X_test_scaled)

print("Naive Bayes Accuracy: ",accuracy_score(y_test, nby_pred))
print("Naive Bayes F1 Score: ",f1_score(y_test, nby_pred, average='weighted'))
print("Naive Bayes Recall Score: ",recall_score(y_test, nby_pred, average='weighted'))
print("Naive Bayes Precision Score: ",precision_score(y_test, nby_pred, average='weighted'))

cm=confusion_matrix(y_test, nby_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Naive Bayes Confusion Matrix')
plt.show()

#Second dataframe
X = df2.drop(columns=['Diabetes_binary'])
Y = df2['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=60)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

nb_model = GaussianNB()
nb_model.fit(X_train_scaled,y_train)
nby_pred = nb_model.predict(X_test_scaled)

print("Naive Bayes Accuracy: ",accuracy_score(y_test, nby_pred))
print("Naive Bayes F1 Score: ",f1_score(y_test, nby_pred, average='weighted'))
print("Naive Bayes Recall Score: ",recall_score(y_test, nby_pred, average='weighted'))
print("Naive Bayes Precision Score: ",precision_score(y_test, nby_pred, average='weighted'))

cm=confusion_matrix(y_test, nby_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Naive Bayes Confusion Matrix')
plt.show()

#Third dataframe
X = df3.drop(columns=['Diabetes_binary'])
Y = df3['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=60)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

nb_model = GaussianNB()
nb_model.fit(X_train_scaled,y_train)
nby_pred = nb_model.predict(X_test_scaled)

print("Naive Bayes Accuracy: ",accuracy_score(y_test, nby_pred))
print("Naive Bayes F1 Score: ",f1_score(y_test, nby_pred, average='weighted'))
print("Naive Bayes Recall Score: ",recall_score(y_test, nby_pred, average='weighted'))
print("Naive Bayes Precision Score: ",precision_score(y_test, nby_pred, average='weighted'))

cm=confusion_matrix(y_test, nby_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Naive Bayes Confusion Matrix')
plt.show()