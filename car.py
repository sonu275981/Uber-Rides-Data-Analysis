import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

taxi_df = pd.read_csv("taxi.csv")
#print(taxi_df.head(5))

data_x = taxi_df.iloc[:,0:-1].values  # selecting all column except last one
data_y = taxi_df.iloc[:,-1].values  # selecting only last column

# print(Data_y)

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 0)

reg = LinearRegression()

reg.fit(X_train,y_train)

print("train score",reg.score(X_train,y_train))
print("test score",reg.score(X_test,y_test))

pickle.dump(reg, open("taxi.pkl", "wb"))  # here wb means write binary

model = pickle.load(open("taxi.pkl", "rb"))  # here wb means read binary

print(model.predict([[80,1770000,6000,85]]))