# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 19:01:57 2018

@author: Alu
"""

Day_before=100
Size_test=100

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

dataset = pd.read_csv('data.csv')
dataset_train = dataset[Size_test:]
dataset_test =  dataset[:Size_test]



################################################# Preprocess Train  ######################################


training_set = dataset_train.iloc[:, 4:5].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(len(training_set)-Day_before):
    X_train.append(training_set_scaled[i+1:i+Day_before +1, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


################################################### Fit Model  #############################################


from sklearn import linear_model
clf2= linear_model.Lasso(alpha=1e-6)
clf2.fit(X_train, y_train)
#clf = linear_model.LassoCV(alphas=np.arange(1e-6,1e-4,1e-7))
#clf.fit(X_train, y_train)

clf2.coef_


############################################### Preprocess Testset  ########################################
    
        
real_stock_price = dataset_test.iloc[:, 4:5].values.reshape(1,-1)[0]
dataset_total = pd.concat((dataset_test['close'] , dataset_train['close']), axis = 0)
inputs = dataset_total[:len(dataset_test) + Day_before].values.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(len(inputs)-Day_before):
    X_test.append(inputs[i+1:i+Day_before+1, 0])
X_test = np.array(X_test)
             
################################################### Predict #################################################
    
predicted_stock_price = clf2.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price.reshape(-1,1)).reshape(1,-1)[0]
        
        
        
############################################## Evalutate ##############################################
        
        
def MSE(y_pred,y_test):
    return (np.sqrt(np.mean((y_pred-y_test)**2)))
        

print(metrics.r2_score(real_stock_price, predicted_stock_price))

############################################## Visualising ############################################



plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


money=1000
for i in range (Size_test-1):
    curent_real = real_stock_price[Size_test-i-1]
    pred = predicted_stock_price[Size_test-i-2]
    next_real = real_stock_price[Size_test-i-2]
    B_pred = (pred-curent_real)/curent_real
    B_real = (next_real-curent_real)/curent_real
    if B_pred*B_real >0 :
        money= money*(1+abs(B_real))
    else :
        money = money*(1-abs(B_real))






