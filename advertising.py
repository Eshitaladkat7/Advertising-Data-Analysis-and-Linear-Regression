import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('advertising.csv')
print(df)

X = df.iloc[:,[0]].values   # : indicates all rows & only column 1 i.e. level
y = df.iloc[:,[3]].values   #  indicates all rows & only column 2 i.e. Salary
print(X)
print(y)


X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
print(X_train,X_test,y_train,y_test)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()      # regressor is our model calling "LinearRegression()"
regressor.fit(X_train,y_train)      # fitting our data in Linear Regression model
y_pred = regressor.predict(X_test)  # making predictions
print(y_test,y_pred)

print('\n\nVariance score: %.2f' % r2_score(y_test,y_pred))   # variance score: 1 is perfect prediction
print('\n Mean Error = ', mean_squared_error(y_test, y_pred)) # mean-squared-error

#plotting the graph for training dataset

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')  #regression line
plt.title("TV vs Sales (Training set)")
plt.xlabel("TV")
plt.ylabel("Sales")
plt.show()

# predicting value of dependent variable (y) for unseen value of independent variable (X)
y_pred = regressor.predict([[11]])
print('\n\n Linear Regression \n Given new X value = 11')
print('Predicted y value = ',y_pred.round(2))