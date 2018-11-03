# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:45:40 2018

@author: Alu
"""

Day_before=200
Size_validation = 1000 
Size_test=100


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv')
dataset_train = dataset[Size_test+Size_validation:]
dataset_validation = dataset[Size_test:Size_test+Size_validation]
dataset_test =  dataset[:Size_test]

def money_score(y_true,y_pred):
    size_test = len(y_true)
    money=1000
    history=[1000]
    for i in range (size_test-1):
        curent_real = y_true[size_test-i-1]
        pred = y_pred[size_test-i-2]
        next_real = y_true[size_test-i-2]
        B_pred = (pred-curent_real)/curent_real
        B_real = (next_real-curent_real)/curent_real
        if B_pred*B_real >0 :
            money = money*(1+abs(B_real))
        else :
            money = money*(1-abs(B_real))
        history.append(money)
    return(np.sum(np.array(history))/1000)


 
    


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


################################################### Fit Models  #############################################



from sklearn import neural_network
        
NN=neural_network.MLPRegressor(hidden_layer_sizes=(100,100,100))
NN.fit(X_train, y_train)


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 30)
rf.fit(X_train, y_train)
   


################################################ Preprocess Testset  ########################################


real_stock_price = dataset_validation.iloc[:, 4:5].values.reshape(1,-1)[0]
dataset_train_vali = pd.concat((dataset_validation['close'] , dataset_train['close']), axis = 0)
inputs = dataset_train_vali[:len(dataset_validation) + Day_before].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_Vali = []
for i in range(len(inputs)-Day_before):
    X_Vali.append(inputs[i+1:i+Day_before+1, 0])
X_Vali = np.array(X_Vali)
             
################################################### Predict #################################################
    
pred_NN = NN.predict(X_Vali)
pred_NN = sc.inverse_transform(pred_NN.reshape(-1,1)).reshape(1,-1)[0]

pred_rf = rf.predict(X_Vali)
pred_rf = sc.inverse_transform(pred_rf.reshape(-1,1)).reshape(1,-1)[0]



##################################################### Strategy #############################################



L=[]
L_acc=[]
grid = np.arange(0,1,0.01)
for a in grid:
    pred = a*pred_NN+(1-a)*pred_rf
    L.append(a)
    L_acc.append(money_score(pred,real_stock_price))
L_acc=np.array(L_acc)
L=np.array(L)
a=L[np.argmax(L_acc)]



# Retrain
training_set = dataset[Size_test:].iloc[:, 4:5].values
training_set_scaled = sc.transform(training_set)
X_train = []
y_train = []
for i in range(len(training_set)-Day_before):
    X_train.append(training_set_scaled[i+1:i+Day_before +1, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)



NN=neural_network.MLPRegressor(hidden_layer_sizes=(150,150,150,150))
NN.fit(X_train, y_train)


rf = RandomForestRegressor(n_estimators = 30)
rf.fit(X_train, y_train)
   

real_stock_price = dataset_test.iloc[:, 4:5].values.reshape(1,-1)[0]
dataset_total = pd.concat((dataset_test['close'] , dataset_validation['close']), axis = 0)
inputs = dataset_total[:len(dataset_test) + Day_before].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(len(inputs)-Day_before):
    X_test.append(inputs[i+1:i+Day_before+1, 0])
X_test = np.array(X_test)
             
  
pred_NN = NN.predict(X_test)
pred_NN = sc.inverse_transform(pred_NN.reshape(-1,1)).reshape(1,-1)[0]
pred_rf = rf.predict(X_test)
pred_rf = sc.inverse_transform(pred_rf.reshape(-1,1)).reshape(1,-1)[0]


final_pred = a*pred_NN+(1-a)*pred_rf


#################################################### Evaluation ############################################

def money(y_pred,y_true):
    size_test = len(y_true)
    money=1000
    for i in range (size_test-1):
        curent_real = y_true[size_test-i-1]
        pred = y_pred[size_test-i-2]
        next_real = y_true[size_test-i-2]
        B_pred = (pred-curent_real)/curent_real
        B_real = (next_real-curent_real)/curent_real
        if B_pred*B_real >0 :
            money = money*(1+abs(B_real))
        else :
            money = money*(1-abs(B_real))
    return(money)


###################################################### Visualisation ##########################################



plt.plot(real_stock_price, color = 'red', label = 'Real')
plt.plot(final_pred, color = 'blue', label = 'prediction finale')
plt.plot(pred_NN, color = 'cyan', label = 'Prediction NN ')
plt.plot(pred_rf, color = 'magenta', label = 'Prediction rf ')


plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


print (money(final_pred,real_stock_price))