# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:04:33 2018

@author: Alu
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model


class Regressor :
    def __init__(self,wallet=1000.0, path = 'data.csv'):
        self.wallet=wallet
        self.path = path
        self.data = 0
        self.scaller = 0
        self.regressor = 0
        self.day_before = 60
        self.previous = 0
        self.next_pred = 0
        
    def ini_data_test(self,i):
        # import data from path with i late day
        self.data=pd.read_csv(self.path)[i:].reset_index(drop=True)
        
    def ini_data(self,equity_name):
        # import data from API in path and update self.data with 
        
        self.data=pd.read_csv(self.path)
        
    def fit(self):
        dataset_train = self.data
        training_set = dataset_train.iloc[:, 4:5].values
        sc = MinMaxScaler(feature_range = (0, 1))
        training_set_scaled = sc.fit_transform(training_set)
        self.scaller = sc
        X_train = []
        y_train = []
        for i in range( len(training_set)-self.day_before):
            X_train.append(training_set_scaled[i+1:i+self.day_before+1, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        clf = linear_model.LassoCV()
        clf.fit(X_train, y_train)
        self.regressor = clf



    def predict (self):
        dataset_total = self.data[:self.day_before+1].reset_index(drop=True)
        inputs = dataset_total.iloc[:, 4:5].values
        inputs = inputs.reshape(-1,1)
        inputs = self.scaller.transform(inputs)
        X_test = []
        for i in range(len(inputs)-self.day_before):
            X_test.append(inputs[i:i+self.day_before, 0])
        X_test = np.array(X_test)
        y_pred = self.regressor.predict(X_test)
        self.next_pred=self.scaller.inverse_transform(y_pred.reshape(-1,1))[0][0]

        
    def get_strat(self):
        current = self.data['close'][0]
        B_pred = (self.next_pred-current)/current
        if B_pred >= 0 :
            print("buy")
        else :
            print ("send")
        self.previous=B_pred
    
    def update_wallet(self):
        current = self.data['close'][0]
        previous = self.data['close'][1]
        B_real = (current-previous)/previous
        if self.previous*B_real >= 0 :
            self.wallet=self.wallet*(1+abs(B_real))
        else :
            self.wallet=self.wallet*(1-abs(B_real))
