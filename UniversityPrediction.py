import pandas as pd
import numpy as np
#import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
df = pd.read_csv("Admission_Predict.csv")
df.drop('Serial No.', axis=1, inplace=True)
x = df.drop(columns = ['Chance of Admit'])
y = df['Chance of Admit']
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
model = LinearRegression()
model.fit(x, y)
#accuracy = model.score(x_test, y_test)
#print(accuracy)
#a = int(input("Your GRE score: \n"))
#b = int(input("Your TOEFL score: \n"))
#c = int(input("The University Rating you want to apply out of 5: \n"))
#d = float(input("Your SOP score at the range of 0 to 5: \n"))
#e = float(input("Your LOR score at the range of 0 to 5: \n"))
#f = float(input("Your CGPA score out of 10: \n"))
#g = input("Have you done any Research (y/n)?: \n")
#if g == 'y':
#   w = 1
#elif g == 'n':
#    w = 0
prediction = model.predict([[321, 105, 4, 4.5, 3.5, 8.50, 0]])
#ans = prediction * 100
#print("The Percentage That you will get Selected in this University is:" + str(ans))
pickle.dump(model, open('University_Prediction.pkl', 'wb'))
