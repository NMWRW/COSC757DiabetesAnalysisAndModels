import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

df = pd.read_csv('Data/diabetes_012_health_indicators_BRFSS2015.csv')

#Linear Regression
X = np.array(df['BMI'])
y = np.array(df['Diabetes_012'])

X=X.reshape(len(X),1)
Y=y.reshape(len(y),1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

X = sm.add_constant(X)  # Adds a constant term to the predictor

bmifit = sm.OLS(y, X).fit()
print(bmifit.summary())

regr = linear_model.LinearRegression()

# Training the model using the training subset
regr.fit(X_train, y_train)

plt.scatter(X_test, y_test)
plt.title('Regression Scatterplot')
plt.xlabel('BMI')
plt.ylabel('Diabetes?')
plt.xticks(())
plt.yticks(())
plt.plot(X_test, regr.predict(X_test), color='black', linewidth=2)

plt.show()

y_pred = regr.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R-squared:', r2)