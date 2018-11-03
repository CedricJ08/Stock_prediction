# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 12:15:45 2018

@author: Alu
"""

Day_before=60
Size_validation = 100 
Size_test=50


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



from sklearn import linear_model
lasso= linear_model.Lasso(alpha=1e-6)


#alphagrid=np.arange(1e-6,1e-4,1e-7)
#len_data = len(X_train)
#len_subdata = len_data//10
#scores=[]
#for i in range (10-2):
#    print(i)
#    X_test_i = X_train[len_subdata*i:len_subdata*(i+1)]
#    y_test_i = training_set.reshape(1,-1)[0][len_subdata*i:len_subdata*(i+1)]
#    X_train_i = X_train[len_subdata*(i+1):min(len_subdata*(i+4),len_data)]
#    y_train_i = y_train[len_subdata*(i+1):min(len_subdata*(i+4),len_data)]
#    scores_i=[]
#    for x in alphagrid :
#        lasso = linear_model.Lasso(alpha=x)
#        lasso.fit(X_train_i, y_train_i)
#        pred = lasso.predict(X_test_i)
#        pred_unscaled = sc.inverse_transform(pred.reshape(-1,1)).reshape(1,-1)[0]
##        scores_i.append(metrics.r2_score(y_test_i,pred))
#        scores_i.append(money_score(y_test_i,pred))
#    scores_i=np.array(scores_i)
#    scores.append(scores_i)
#scores = np.array(scores)
#final_scores = sum(scores)/len(scores)
#alpha_opti=alphagrid[np.argmax(final_scores)]     
#lasso = linear_model.Lasso(alpha=alpha_opti)

lasso.fit(X_train, y_train)     



from sklearn.svm import SVR
svr=SVR(C=0.41,kernel = 'rbf',gamma=0.03)

#Cgrid=np.arange(0.01,2,0.04)
#gammagrid=np.arange(0.001,0.35,0.005)
#
#len_data = len(X_train)
#len_subdata = len_data//10
#scores=[]
#for i in range (10-3):
#    print (i)
#    X_test_i = X_train[len_subdata*i:len_subdata*(i+1)]
#    y_test_i = training_set.reshape(1,-1)[0][len_subdata*i:len_subdata*(i+1)]
#    X_train_i = X_train[len_subdata*(i+1):min(len_subdata*(i+4),len_data)]
#    y_train_i = y_train[len_subdata*(i+1):min(len_subdata*(i+4),len_data)]
#    scores_i=[]
#    for x in gammagrid :
#        scores_i_x=[]
#        for y in Cgrid :
#            svr = SVR(C=y,kernel = 'rbf',gamma=x)
#            svr.fit(X_train_i, y_train_i)
#            pred = svr.predict(X_test_i)            
#            pred_unscaled = sc.inverse_transform(pred.reshape(-1,1)).reshape(1,-1)[0]
##            scores_i_x.append(metrics.r2_score(y_test_i,pred))
#            scores_i_x.append(money_score(y_test_i,pred_unscaled))
#        scores_i_x=np.array(scores_i_x)
#        scores_i.append(scores_i_x)
#    scores_i=np.array(scores_i)
#    scores.append(scores_i)
#scores = np.array(scores)
#final_scores = sum(scores)/len(scores)
#index_max = np.unravel_index(final_scores.argmax(), final_scores.shape)
#gamma_opti=gammagrid[index_max[0]]  
#C_opti= Cgrid[index_max[1]]
#svr = SVR( C=C_opti, kernel = 'rbf', gamma=gamma_opti )

svr.fit(X_train, y_train)




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
    
pred_lasso = lasso.predict(X_Vali)
pred_lasso = sc.inverse_transform(pred_lasso.reshape(-1,1)).reshape(1,-1)[0]
pred_svr = svr.predict(X_Vali)
pred_svr = sc.inverse_transform(pred_svr.reshape(-1,1)).reshape(1,-1)[0]
pred_rf = rf.predict(X_Vali)
pred_rf = sc.inverse_transform(pred_rf.reshape(-1,1)).reshape(1,-1)[0]



##################################################### Strategy #############################################



L=[]
L_acc=[]
grid = np.arange(0,1,0.01)
for i in grid:
    grid_i= np.arange(0,1-i,0.01)
    for j in grid_i:
        pred = i*pred_lasso+j*pred_svr+(1-i-j)*pred_rf
        L.append([i,j])
        L_acc.append(money_score(pred,real_stock_price))
L_acc=np.array(L_acc)
L=np.array(L)
a1=L[np.argmax(L_acc)][0]
a2=L[np.argmax(L_acc)][1]


# Retrain
training_set = dataset[Size_test:].iloc[:, 4:5].values
training_set_scaled = sc.transform(training_set)
X_train = []
y_train = []
for i in range(len(training_set)-Day_before):
    X_train.append(training_set_scaled[i+1:i+Day_before +1, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)




lasso = linear_model.LassoCV(alphas=np.arange(1e-6,1e-4,1e-7))
lasso.fit(X_train, y_train)

svr = SVR(C=20,kernel = 'rbf',gamma=0.1)
svr.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators = 30, random_state = 0)
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
             
  
pred_lasso = lasso.predict(X_test)
pred_lasso = sc.inverse_transform(pred_lasso.reshape(-1,1)).reshape(1,-1)[0]
pred_svr = svr.predict(X_test)
pred_svr = sc.inverse_transform(pred_svr.reshape(-1,1)).reshape(1,-1)[0]
pred_rf = rf.predict(X_test)
pred_rf = sc.inverse_transform(pred_rf.reshape(-1,1)).reshape(1,-1)[0]


final_pred = a1*pred_lasso+a2*pred_svr+(1-a1-a2)*pred_rf


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
plt.plot(pred_lasso, color = 'cyan', label = 'Prediction Lasso ')
plt.plot(pred_rf, color = 'magenta', label = 'Prediction rf ')
plt.plot(pred_svr, color = 'yellow', label = 'Prediction svr ')

plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


print (money(final_pred,real_stock_price))