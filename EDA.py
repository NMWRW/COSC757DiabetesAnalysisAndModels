import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tabulate import tabulate



def countClasses(df):
    print("Class distribution:")
    class_column = df.columns[0]
    class_counts = df[class_column].value_counts()
    for class_value, count in class_counts.items():
        print(f"Class {class_value}: {count} instances")
 
def Analysis(df,dataset):
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
    print(f"\n{'='*80}")
    print(f"Descriptive Statistics for {dataset} Dataset")
    print(f"{'='*80}\n")

    desc = df.describe().transpose().round(3)
    print(tabulate(desc, headers='keys', tablefmt='github'))
    print("\n")  # blank line for spacing

    correlation_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix for ' + dataset + ' Dataset')
    plt.xticks(rotation=30, ha='right')
    plt.show()

    numeric_cols = df.select_dtypes(include='number').columns

    
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        unique_values = df[col].dropna().unique()
        num_unique = len(unique_values)

        if num_unique <= 15:
            # --- Discrete or Binary Feature: one bar per unique value ---
            counts = df[col].value_counts().sort_index()
            sns.barplot(x=counts.index.astype(str), y=counts.values, palette="crest")
            plt.title(f"{col} for {dataset} Dataset")
            plt.xlabel("Value")
            plt.ylabel("Count")
            
        
        else:
            # --- Continuous variable ---
            min_val = int(np.floor(df[col].min()))
            max_val = int(np.ceil(df[col].max()))
            data_range = max_val - min_val

            # Choose number of bins (up to 20, but fewer if small range)
            num_bins = min(20, data_range)
            if num_bins < 5:  # ensure at least a few bins
                num_bins = data_range if data_range > 0 else 1

            # Create equal-width integer bins
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)

            sns.histplot(df[col],bins=bin_edges,kde=False,color='skyblue',edgecolor='black')
            plt.title(f"{col} (Continuous) for {dataset} Dataset")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.xlim(min_val, max_val)

        plt.tight_layout()
        plt.show()
        
df = pd.read_csv('Data/diabetes_012_health_indicators_BRFSS2015.csv')
Analysis(df, "012_health _indicators")

df = pd.read_csv('Data/diabetes_binary_health_indicators_BRFSS2015.csv')
Analysis(df,"Binary")

df = pd.read_csv('Data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
Analysis(df,"Binary 50/50 split")