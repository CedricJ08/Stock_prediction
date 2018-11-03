# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:02:45 2018

@author: Alu
"""

Day_before= 60
Size_test = 100   
    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('data.csv')
dataset_train = dataset[Size_test:]
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
            
            
################################################### Fit Model  #############################################
        
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
        
################################################# Preprocess Testset  ########################################
                
                    
real_stock_price = dataset_test.iloc[:, 4:5].values
dataset_total = pd.concat((dataset_test['close'] , dataset_train['close']), axis = 0)
inputs = dataset_total[:len(dataset_test) + Day_before].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(len(inputs)-Day_before):
    X_test.append(inputs[i+1:i+Day_before+1, 0])
X_test = np.array(X_test)
                         
################################################### Predict #################################################
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price).reshape(1,-1)[0]
          
############################################## Evalutate ##############################################
            
      
y_pred=np.load('pred_60.npy')
y_true = np.load('true_60.npy')
     
#def MSE(y_pred,y_test):
#    return (np.sqrt(np.mean((y_pred-y_test)**2)))
            
    
#print(metrics.r2_score(real_stock_price, predicted_stock_price))

############################################## Visualising ############################################


plt.plot(y_true, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


money=1000
for i in range (Size_test-1):
    curent_real = y_true[Size_test-i-1]
    pred = y_pred[Size_test-i-2]
    next_real = y_true[Size_test-i-2]
    B_pred = (pred-curent_real)/curent_real
    B_real = (next_real-curent_real)/curent_real
    if B_pred*B_real >0 :
        money = money*(1+abs(B_real))
    else :
        money = money*(1-abs(B_real))
    print (money)




    
#print(money_score(real_stock_price,predicted_stock_price))