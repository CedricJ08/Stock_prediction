# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:13:41 2018

@author: Alu
"""
import numpy as np
import matplotlib.pyplot as plt
from Lasso_object import Regressor
visu_pred =[]
x=[]
visu_real=[]
def main():
    R=Regressor()
    test=100

    for i in range (test):
        x.append(i)
        R.ini_data_test(test-1-i)
        if i !=0 :
            R.update_wallet()
            print(R.wallet)
            visu_real.append(R.data['close'][0])
            
        
        R.fit()
        R.predict()
        R.get_strat()
        visu_pred.append(R.next_pred)


main()
visu_pred.pop()
x.pop()



plt.plot(x,visu_pred)
plt.plot(x,visu_real)

np.sqrt(np.mean((np.array(visu_pred)-np.array(visu_real))**2))