# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:46:07 2018

@author: Alu
"""
import pickle
#import requests
#import csv
#import pandas as pd
#import time
import numpy as np
#from numpy import unravel_index

#####################################################" Import Data #######################################"

#L_index=['ATVI','ADBE','ALXN','GOOGL','ALGN','GOOG','AMZN','AAL','AMGN','ADI','AAPL','AMAT','ASML','ADSK','ADP','BIDU','BIIB','BMRN','BKNG','AVGO','CA','CDNS','CELG','CERN','CHTR','CHKP','CTAS','CSCO','CTXS','CTSH','CMCSA','COST','CSX','CTRP','XRAY','DLTR','EBAY','EA','EXPE','ESRX','FB','FAST','FISV','GILD','HAS','HSIC','HOLX','IDXX','ILMN','INCY','INTC','INTU','ISRG','JBHT','JD','KLAC','LRCX','LBTYA','LBTYK','MAR','MXIM','MELI','MCHP','MU','MSFT','MDLZ','MNST','MYL','NTES','NFLX','NVDA','ORLY','PCAR','PAYX','PYPL','PEP','QCOM','QRTEA','REGN','ROST','STX','SHPG','SIRI','SWKS','SBUX','SYMC','SNPS','TMUS','TTWO','TSLA','TXN','KHC','FOXA','FOX','ULTA','VRSK','VRTX','VOD','WBA','WDC','WDAY','WYNN','XLNX']

#i=0
#for index in ['XLNX']:
#    print("process : "+str(i)+"%")
#    remaning_time_min= 15 * (103-i)//60
#    print("remaning time : "+str(remaning_time_min)+"mins")
#    i+=1
#    time.sleep(15)
#    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&outputsize=full&datatype=csv&apikey=ACAZK7H8AYZVKZBV'.format(index)
#    data = requests.get(url)
#    file_name='data{}.csv'.format(index) 
#    with open(file_name, 'w') as f:
#        writer = csv.writer(f)
#        reader = csv.reader(data.text.splitlines())
#    
#        for row in reader:
#            writer.writerow(row)


####################################################### Preprocessing ################################

#df_dic={}
#for index in L_index:
#    file_name='data{}.csv'.format(index) 
#    df_dic[index]=pd.read_csv(file_name)
#
#L_smal=[] 
#for i in df_dic:
#    print(len(df_dic[i]))
#    if len(df_dic[i])<4000:
#        L_smal.append(i)
#
#for i in L_smal:
#    del df_dic[i]


#f = open("dic_df.pkl","wb")
#pickle.dump(df_dic,f)
#f.close()

#L=[]
#for i in df_dic :
#    L.append(df_dic[i][:4000])

#name_ind=[]
#for i in df_dic:
#    name_ind.append(i)
#name_ind=np.array(name_ind)


f= open("dic_df.pkl", 'rb') 
df_dic=pickle.load(f)

######################################################## Corr Matrice #####################################


#l=[] 
#for i in range(len(L)):
#    l.append(np.array(L[i]['close']))

#l=np.array(l)
#corr_mat=np.corrcoef(l)

#np.save('name_ind',name_ind)
#np.save('correlation_matrice',corr_mat)


name_ind=np.load('name_ind.npy')
corr_mat=np.load('correlation_matrice.npy')



##################################################### More Corolated ####################################

#corr_abso=np.absolute(corr_mat)
#corr_copy=np.copy(corr_abso)
#
#
#triangle_corr=np.zeros(np.shape(corr_copy))
#for i in range (len(corr_copy)):
#    for j in range (i):
#        triangle_corr[i][j]=corr_copy[i][j]
#        
#liste_most_cor_ind = []
#for i in range (20):
#    ind=unravel_index(triangle_corr.argmax(), triangle_corr.shape)
#    liste_most_cor_ind.append(ind)
#    triangle_corr[ind] = 0
#
#
#liste_most_cor_name=[]
#for i in liste_most_cor_ind :
#    liste_most_cor_name.append((name_ind[i[0]],name_ind[i[1]]))
#
#
#np.save('most_cor_ind',np.array(liste_most_cor_ind))
#np.save('most_cor_name',np.array(liste_most_cor_name))

liste_most_cor_ind=np.load('most_cor_ind.npy')
liste_most_cor_name=np.load('most_cor_name.npy')
       
#################################################### Less corrolated ##########################################


#corr_abso=np.absolute(corr_mat)
#corr_copy=np.copy(corr_abso)
#
#
#triangle_corr=np.ones(np.shape(corr_copy))
#for i in range (len(corr_copy)):
#    for j in range (i):
#        triangle_corr[i][j]=corr_copy[i][j]
#        
#liste_less_cor_ind = []
#for i in range (20):
#    ind=unravel_index(triangle_corr.argmin(), triangle_corr.shape)
#    liste_less_cor_ind.append(ind)
#    triangle_corr[ind] = 1
#
#
#liste_less_cor_name=[]
#for i in liste_less_cor_ind :
#    liste_less_cor_name.append((name_ind[i[0]],name_ind[i[1]]))
#
#
#np.save('less_cor_ind',np.array(liste_less_cor_ind))
#np.save('less_cor_name',np.array(liste_less_cor_name))


liste_less_cor_ind=np.load('less_cor_ind.npy')
liste_less_cor_name=np.load('less_cor_name.npy')
 

###################################################### Smaller Beta ###############################################



test_df=df_dic['ATVI']

test_df_close_2017=[]
test_df_date_2017=[]
for i in range (len(test_df)):
    if test_df['timestamp'][i][0:4]=='2017':
        test_df_close_2017.append(test_df['close'][i])
        test_df_date_2017.append(test_df['timestamp'][i])

test_df_close_2016=[]
test_df_date_2016=[]
for i in range (len(test_df)):
    if test_df['timestamp'][i][0:4]=='2016':
        test_df_close_2016.append(test_df['close'][i])
        test_df_date_2016.append(test_df['timestamp'][i])

close_2017=[]
close_2016=[]
for i in range (len(test_df_close_2017)):
    for j in range (len(test_df_close_2016)):

        if test_df_date_2017[i][5:]==test_df_date_2016[j][5:]:
            close_2017.append(test_df_close_2017[i])
            close_2016.append(test_df_close_2016[j])


