# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:33:30 2018

@author: Alu
"""

Day_before= 60
Size_test = 100   
    
import numpy as np
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
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

predicted_stock_price=predicted_stock_price.reshape(1,-1)[0]
real_stock_price = real_stock_price.reshape(1,-1)[0]

np.save('pred',predicted_stock_price)
np.save('true',real_stock_price)