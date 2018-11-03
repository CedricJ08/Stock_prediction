# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 12:29:23 2018

@author: Alu
"""

Day_before=200
Size_test=100

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


from sklearn import neural_network
NN=neural_network.MLPRegressor(hidden_layer_sizes=(150,150,150,150))
NN.fit(X_train, y_train)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 30)
rf.fit(X_train, y_train)


############################################### Preprocess Testset  ########################################
    
        
real_stock_price = dataset_test.iloc[:, 4:5].values.reshape(1,-1)[0]
dataset_total = pd.concat((dataset_test['close'] , dataset_train['close']), axis = 0)
inputs = dataset_total[:len(dataset_test) + Day_before].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(len(inputs)-Day_before):
    X_test.append(inputs[i+1:i+Day_before+1, 0])
X_test = np.array(X_test)
             
################################################### Predict #################################################
    
pred_NN = NN.predict(X_test)
pred_NN = sc.inverse_transform(pred_NN.reshape(-1,1)).reshape(1,-1)[0]
pred_rf = rf.predict(X_test)
pred_rf = sc.inverse_transform(pred_rf.reshape(-1,1)).reshape(1,-1)[0]

  

        
############################################## Strategy ##############################################


money=1000
for i in range (Size_test-1):
    curent_real = real_stock_price[Size_test-i-1]
    predrf = pred_rf[Size_test-i-2]
    predNN = pred_NN[Size_test-i-2]
    next_real = real_stock_price[Size_test-i-2]
    B_predrf = (predrf-curent_real)/curent_real
    B_predNN = (predNN-curent_real)/curent_real
    B_real = (next_real-curent_real)/curent_real
    if np.sign(B_predrf)==np.sign(B_predNN):
        if B_predrf*B_real >= 0 :
            money= money*(1+abs(B_real))
        else :
            money = money*(1-abs(B_real))


money_rf_seul=1000
for i in range (Size_test-1):
    curent_real = real_stock_price[Size_test-i-1]
    predrf = pred_rf[Size_test-i-2]
    next_real = real_stock_price[Size_test-i-2]
    B_predrf = (predrf-curent_real)/curent_real
    B_real = (next_real-curent_real)/curent_real
    if B_predrf*B_real >= 0 :
        money_rf_seul= money_rf_seul*(1+abs(B_real))
    else :
        money_rf_seul = money_rf_seul*(1-abs(B_real))

money_NN_seul=1000
for i in range (Size_test-1):
    curent_real = real_stock_price[Size_test-i-1]
    predNN = pred_NN[Size_test-i-2]
    next_real = real_stock_price[Size_test-i-2]
    B_predNN = (predNN-curent_real)/curent_real
    B_real = (next_real-curent_real)/curent_real
    if B_predNN*B_real >= 0 :
        money_NN_seul= money_NN_seul*(1+abs(B_real))
    else :
        money_NN_seul = money_NN_seul*(1-abs(B_real))     

print('money_duo : '+str(money))
print('money_rf : '+str(money_rf_seul))
print('money_NN : '+str(money_NN_seul))

############################################## Visualising ############################################



#plt.plot(real_stock_price, color = 'red', label = 'Real')
#plt.plot(pred_rf, color = 'magenta', label = 'Prediction rf ')
#plt.plot(pred_NN, color = 'yellow', label = 'Prediction NN ')
#plt.title('Stock Price Prediction')
#plt.xlabel('Time')
#plt.ylabel('Stock Price')
#plt.legend()
#plt.show()




