import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np



def countClasses(df):
    print("Class distribution:")
    class_column = df.columns[0]
    class_counts = df[class_column].value_counts()
    for class_value, count in class_counts.items():
        print(f"Class {class_value}: {count} instances")
 
def Analysis(df):
    #Quick clean 
    #Drop nulls
    df.dropna(inplace=True)
    #Drop duplicates
    df.drop_duplicates(inplace=True)
    
    countClasses(df)
    
    print("Sample data")
    print(df.sample(5)) 
    print("Data info")     
    df.info()     
    print("describe")          
    print(df.describe())

    correlation_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix')
    plt.xticks(rotation=30, ha='right')
    plt.show()

    # Plot distributions for all numeric columns
    for col in df.select_dtypes(include='number').columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], bins=30, kde=True, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
    

    

#corelation matrix 
df = pd.read_csv('Data/diabetes_012_health_indicators_BRFSS2015.csv')
countClasses(df)
df = pd.read_csv('Data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
countClasses(df)
df = pd.read_csv('Data/diabetes_binary_health_indicators_BRFSS2015.csv')
countClasses(df)