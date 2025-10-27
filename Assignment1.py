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


df = pd.read_csv('wine+quality/winequality-white.csv')
columns_of_interest = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

def Analysis():
    print(df.sample(5))      
    df.info()     
    print("describe")          
    print(df.describe())

    # Calculate the correlation matrix
    correlation_matrix = df[columns_of_interest].corr()
    print(correlation_matrix)
    # Display the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix')
    plt.xticks(rotation=30, ha='right')
    plt.show()
    for col in columns_of_interest:
        plt.figure(figsize=(10,5))
        sns.histplot(df[col], bins=30, kde=True, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()





def minMaxScaling(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled['quality'] = y.values
    return df_scaled

def zScoreScaling(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled['quality'] = y.values
    return df_scaled

def decimalScaling(X, y):
    df_scaled = X.copy()
    for col in df_scaled.columns:
        max_abs_val = df_scaled[col].abs().max()
        j = np.ceil(np.log10(max_abs_val + 1))
        df_scaled[col] = df_scaled[col] / (10 ** j)
    df_scaled['quality'] = y.values
    return df_scaled

def ScalingForPrint():
    X = df.drop(columns=['quality'])
    y = df['quality']

    # Min-Max Scaling
    df_scaled_minMax = minMaxScaling(X, y)
    print("Min-Max Scaled Data")
    print(df_scaled_minMax.head())
    
    # Z-Score Scaling
    df_scaled_z = zScoreScaling(X, y)
    print("Standard (Z-Score) Scaled Data")
    print(df_scaled_z.head())

    # Decimal Scaling
    df_scaled_decimal = decimalScaling(X, y)
    print("Decimal Scaled Data")
    print(df_scaled_decimal.head())
    
    return df_scaled_minMax, df_scaled_z, df_scaled_decimal

def BinningByClustering(df):
# binning alcohol levels
    bins = 5
    X = df[['alcohol']].values
    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='kmeans')
    est.fit(X)
    X_binned = est.transform(X)
    df_binned = pd.DataFrame(X_binned, columns=['alcohol_binned'])
    print("Binned Alcohol Levels")
    print(df_binned.head())
    bin_counts = df_binned['alcohol_binned'].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(8,5))
    sns.barplot(x=bin_counts.index.astype(int), y=bin_counts.values, palette='Blues_d')
    plt.title('Alcohol Levels Binned Clustering (Min-Max Scaled)')
    plt.xlabel('Bin Number')
    plt.ylabel('Number of Samples')
    plt.show()

def BinningByEqualFreq(df):
    # binning alcohol levels
    bins = 5
    X = df[['alcohol']].values
    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
    est.fit(X)
    X_binned = est.transform(X)
    df_binned = pd.DataFrame(X_binned, columns=['alcohol_binned'])
    print("Binned Alcohol Levels")
    print(df_binned.head())
    bin_counts = df_binned['alcohol_binned'].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(8,5))
    sns.barplot(x=bin_counts.index.astype(int), y=bin_counts.values, palette='Blues_d')
    plt.title('Alcohol Levels Binned Equal Frequency (Min-Max Scaled)')
    plt.xlabel('Bin Number')
    plt.ylabel('Number of Samples')
    plt.show()
    
def NormalizingSkewLog(df,column):
    data = df[column].values
    data_log = np.log(data)
    fig, axs = plt.subplots(nrows=1, ncols=2)
    #create histograms
    axs[0].hist(data, edgecolor='black')
    axs[1].hist(data_log, edgecolor='black')

    #add title to each histogram
    fig.suptitle(f'Log Normalization of {column}', fontsize=14, fontweight='bold')
    axs[0].set_title('Original Data')
    axs[1].set_title('Log-Transformed Data')
    plt.show()
    return data_log

def NormalizingSkewSq(df,column):
    data = df[column].values
    data_sqrt = np.sqrt(data)
    fig, axs = plt.subplots(nrows=1, ncols=2)
    #create histograms
    axs[0].hist(data, edgecolor='black')
    axs[1].hist(data_sqrt, edgecolor='black')

    #add title to each histogram
    fig.suptitle(f'Square-Root Normalization of {column}', fontsize=14, fontweight='bold')
    axs[0].set_title('Original Data')
    axs[1].set_title('Square-Root Transformed Data')
    plt.show()
    return data_sqrt

def NormalizingSkewInvSqrt(df,column):
    data = df[column].values
    data_invsqrt = 1 / np.sqrt(data)
    fig, axs = plt.subplots(nrows=1, ncols=2)
    #create histograms
    axs[0].hist(data, edgecolor='black')
    axs[1].hist(data_invsqrt, edgecolor='black')

    #add title to each histogram
    fig.suptitle(f'Inverse Square-Root Normalization of {column}', fontsize=14, fontweight='bold')
    axs[0].set_title('Original Data')
    axs[1].set_title('Inverse Square-Root Transformed Data')
    plt.show()
    return data_invsqrt

    
df_scaled_minMax, df_scaled_z, df_scaled_decimal = ScalingForPrint()

Analysis() 

BinningByClustering(df)
BinningByEqualFreq(df)

DataLog = NormalizingSkewLog(df,'residual sugar')
DataSqrt = NormalizingSkewSq(df,'residual sugar')
dataInvSqrt = NormalizingSkewInvSqrt(df,'residual sugar')

DataLog = NormalizingSkewLog(df,'alcohol')
DataSqrt = NormalizingSkewSq(df,'alcohol')
dataInvSqrt = NormalizingSkewInvSqrt(df,'alcohol')

X_scaled = df_scaled_minMax.drop(columns=['quality'])
y = df['alcohol']

X_scaled['quality'] = df['quality'].values

features = ['density', 'residual sugar',  'quality', 'pH', "fixed acidity"]
X = X_scaled[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print("Linear Regression (Scaled Features)")
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print(f"Root mean squared error: {rmse:.4f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='skyblue', edgecolor='k', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual Alcohol')
plt.ylabel('Predicted Alcohol')
plt.title('Linear Regression (Scaled Features): Actual vs Predicted Alcohol')
plt.grid(True)
plt.show()