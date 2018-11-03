# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:07:11 2018

@author: Alu
"""
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import metrics


############################################## load pred ############################################



real_stock_price=np.load('y_pred_10_4layers_100input.npy')
predicted_stock_price=np.load('y_pred_10_4layers_100input.npy')


############################################## Evaluate Regression ############################################



#def MSE(y_pred,y_test):
#    return (np.sqrt(np.mean((y_pred-y_test)**2)))
#            
#print(metrics.r2_score(real_stock_price, predicted_stock_price))  
#
#
#
############################################### Evaluate Gain ############################################
#
#
#
#
#
##money=1000
##for i in range (Size_test-1):
##    curent_real = real_stock_price[Size_test-i-1]
##    pred = predicted_stock_price[Size_test-i-2]
##    next_real = real_stock_price[Size_test-i-2]
##    B_pred = (pred-curent_real)/curent_real
##    B_real = (next_real-curent_real)/curent_real
##    if B_pred*B_real >0 :
##        money = money*(1+abs(B_real))
##    else :
##        money = money*(1-abs(B_real))
##    print (money)
#
#
#def money_score(y_true,y_pred):
#    size_test = len(y_true)
#    money=1000
#    history=[1000]
#    for i in range (size_test-1):
#        curent_real = y_true[size_test-i-1]
#        pred = y_pred[size_test-i-2]
#        next_real = y_true[size_test-i-2]
#        B_pred = (pred-curent_real)/curent_real
#        B_real = (next_real-curent_real)/curent_real
#        if B_pred*B_real >0 :
#            money = money*(1+abs(B_real))
#        else :
#            money = money*(1-abs(B_real))
#        history.append(money)
#    return(np.sum(np.array(history))/1000)
#
#
#    
#print(money_score(real_stock_price,predicted_stock_price))


############################################## Visualising ############################################



plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


