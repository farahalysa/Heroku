#library
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

#dataset 
df = pd.read_csv("tv_lg.csv", header=0)

df['Layar'] = pd.to_numeric(df['Layar'],errors='coerce')
df['Harga'] = pd.to_numeric(df['Harga'],errors='coerce')

X= df['Layar'].values.reshape(-1, 1)
Y= df['Harga'].values

#call model regression
model = LinearRegression().fit(X,Y)

#save model
filename = 'model.sav'
joblib.dump(model, filename)

#load model
loaded_model = joblib.load(filename)

#prediction model
loaded_model.predict([[20]])