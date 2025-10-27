import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score





def Analysis(df):
    #Quick clean 
    #Drop nulls
    df.dropna(inplace=True)
    #Drop duplicates
    df.drop_duplicates(inplace=True)
    
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

    '''
    # Plot distributions for all numeric columns
    for col in df.select_dtypes(include='number').columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], bins=30, kde=True, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
    '''
# Run analysis

#corelation matrix 
df = pd.read_csv('Data/diabetes_012_health_indicators_BRFSS2015.csv')
Analysis(df)
df = pd.read_csv('Data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
Analysis(df)
df = pd.read_csv('Data/diabetes_binary_health_indicators_BRFSS2015.csv')
Analysis(df)