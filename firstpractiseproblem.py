import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df=pd.read_csv("homeprices.csv")
df
plt.scatter(df.area,df.price,color="red",marker="+")

reg=linear_model.LinearRegression()
reg.fit(df[["area"]],df.price)
reg.predict([[3300]])


#File 2

p=pd.read_csv("areas.csv")
p
d=reg.predict(p)
d
p["PredictedPrice"]=d
p.to_csv("Predict.csv")

