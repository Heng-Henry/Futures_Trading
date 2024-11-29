# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:47:16 2020

@author: NAVY
"""
import numpy as np
import numpy.matlib
from scipy.linalg import eigh
from scipy.linalg import orth


def JCItestpara(X_data,model_type,lag_p):
    if model_type == 'model1':
        model_type = 1
    elif model_type == 'model2':
        model_type = 2
    elif model_type == 'model3':
        model_type = 3
    [NumObs,NumDim] = X_data.shape

    dY_ALL = X_data[1:, :] - X_data[0:-1, :] 
    dY = dY_ALL[lag_p:, :] #DY
    Ys = X_data[lag_p:-1, :] #Lag_Y
    
    #底下開始處理估計前的截距項與時間趨勢項
    if lag_p == 0:
        if model_type == 1:
            dX = np.zeros([NumObs-1, NumDim]) #DLag_Y
        elif model_type == 2:
            dX = np.zeros([NumObs-1, NumDim]) #DLag_Y
            Ys = np.hstack( ( Ys, np.ones((NumObs-lag_p-1,1)) ) )#Lag_Y
        elif model_type == 3:
            dX = np.ones((NumObs-lag_p-1,1)) #DLag_Y
        elif model_type == 4:
            dX = np.ones((NumObs-lag_p-1,1)) #DLag_Y
            Ys = np.hstack( ( Ys, np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )#Lag_Y
        elif model_type == 5:
            dX = np.hstack( ( np.ones((NumObs-lag_p-1,1)) , np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )
    
    elif lag_p>0:
        dX = np.zeros([NumObs-lag_p-1, NumDim * lag_p]) #DLag_Y
        for xi in range(lag_p):
            dX[:, xi * NumDim:(xi + 1) * NumDim] = dY_ALL[lag_p - xi -1 :NumObs - xi - 2, :]
        if model_type == 2:
            Ys = np.hstack( ( Ys, np.ones((NumObs-lag_p-1,1)) ) )
        elif model_type == 3:
            dX = np.hstack( ( dX, np.ones((NumObs-lag_p-1,1)) ) )
        elif model_type == 4:
            Ys = np.hstack( ( Ys, np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )
            dX = np.hstack( ( dX, np.ones((NumObs-lag_p-1,1)) ) )
        elif model_type == 5:
            dX = np.hstack( ( dX, np.ones((NumObs-lag_p-1,1)) , np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )
    
    # 準備開始估計，先轉成matrix，計算比較直觀
    dX, dY, Ys = np.mat(dX), np.mat(dY), np.mat(Ys)

    # 先求dX'*dX 方便下面做inverse
    dX_2 = dX.T * dX
    # I-dX * (dX'*dX)^-1 * dX'
    #python無法計算0矩陣的inverse，用判斷式處理
    if  np.sum(dX_2) == 0:
        M = np.identity(NumObs-lag_p-1) - dX * dX.T
    else:
        M = np.identity(NumObs-lag_p-1) - dX * dX_2.I * dX.T
    
    R0, R1 = dY.T * M, Ys.T * M
    
    S00 = R0 * R0.T / (NumObs-lag_p-1)
    S01 = R0 * R1.T / (NumObs-lag_p-1)
    S10 = R1 * R0.T / (NumObs-lag_p-1)
    S11 = R1 * R1.T / (NumObs-lag_p-1)
    
    #計算廣義特徵值與廣義特徵向量
    eigValue_lambda, eigvecs = eigh(S10 * S00.I * S01, S11, eigvals_only=False)
    
    # 排序特徵向量Eig_vector與特徵值lambda
    sort_ind = np.argsort(-eigValue_lambda)
    #eigValue_lambda = eigValue_lambda[sort_ind]
    eigVecs = eigvecs[:, sort_ind]
    #eigValue_lambda = eigValue_lambda.reshape( len(eigValue_lambda) , 1)
    
    #Beta
    jci_beta = np.mat(eigVecs[:,0][0:2])
    jci_beta = jci_beta.T
    #Alpha
    a = np.mat(eigVecs[:,0])
    jci_alpha = S01 * a.T
    #初始化 c0, d0, c1, d1
    c0 , d0 = 0, 0
    c1 , d1 = np.zeros([NumDim, 1]), np.zeros([NumDim, 1])
    
    #計算 c0, d0, c1, d1，與殘差及VEC項的前置
    if model_type == 1:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        
    elif model_type == 2:
        c0 = eigVecs[-1, 0:1]
        W = dY - (Ys[:,0:2] * jci_beta + numpy.matlib.repmat(c0, NumObs-lag_p-1, 1) )* jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        
    elif model_type == 3:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W
        P = P.T
        c = P[:,-1]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
    
    elif model_type == 4:
        d0 = eigVecs[-1, 0:1]
        W = dY - (Ys[:,0:2] * jci_beta + np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) * d0) * jci_alpha.T
        P = dX.I * W
        P = P.T
        c = P[:,-1]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        
    elif model_type == 5:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        c = P[:,-2]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        d = P[:,-1]
        d0 = jci_alpha.I * d
        d1 = d - jci_alpha * d0
    
    #計算殘差    
    ut = W - dX * P.T
    
    #計算VEC項
    gamma = []
    for bi in range(1,lag_p+1):
        Bq = P[:, (bi-1)*NumDim : bi * NumDim]
        gamma.append(Bq)
    
    #把Ct統整在一起
    Ct = jci_alpha*c0 + c1 + jci_alpha*d0 +d1
    
    return jci_alpha, jci_beta, Ct, ut, gamma

def JCItestpara_spilCt(X_data,model_type,lag_p):
    if model_type == 'model1':
        model_type = 1
    elif model_type == 'model2':
        model_type = 2
    elif model_type == 'model3':
        model_type = 3
    [NumObs,NumDim] = X_data.shape

    dY_ALL = X_data[1:, :] - X_data[0:-1, :] 
    dY = dY_ALL[lag_p:, :] #DY
    Ys = X_data[lag_p:-1, :] #Lag_Y
    
    #底下開始處理估計前的截距項與時間趨勢項
    if lag_p == 0:
        if model_type == 1:
            dX = np.zeros([NumObs-1, NumDim]) #DLag_Y
        elif model_type == 2:
            dX = np.zeros([NumObs-1, NumDim]) #DLag_Y
            Ys = np.hstack( ( Ys, np.ones((NumObs-lag_p-1,1)) ) )#Lag_Y
        elif model_type == 3:
            dX = np.ones((NumObs-lag_p-1,1)) #DLag_Y
        elif model_type == 4:
            dX = np.ones((NumObs-lag_p-1,1)) #DLag_Y
            Ys = np.hstack( ( Ys, np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )#Lag_Y
        elif model_type == 5:
            dX = np.hstack( ( np.ones((NumObs-lag_p-1,1)) , np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )
    
    elif lag_p>0:
        dX = np.zeros([NumObs-lag_p-1, NumDim * lag_p]) #DLag_Y
        for xi in range(lag_p):
            dX[:, xi * NumDim:(xi + 1) * NumDim] = dY_ALL[lag_p - xi -1 :NumObs - xi - 2, :]
        if model_type == 2:
            Ys = np.hstack( ( Ys, np.ones((NumObs-lag_p-1,1)) ) )
        elif model_type == 3:
            dX = np.hstack( ( dX, np.ones((NumObs-lag_p-1,1)) ) )
        elif model_type == 4:
            Ys = np.hstack( ( Ys, np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )
            dX = np.hstack( ( dX, np.ones((NumObs-lag_p-1,1)) ) )
        elif model_type == 5:
            dX = np.hstack( ( dX, np.ones((NumObs-lag_p-1,1)) , np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )
    
    # 準備開始估計，先轉成matrix，計算比較直觀
    dX, dY, Ys = np.mat(dX), np.mat(dY), np.mat(Ys)

    # 先求dX'*dX 方便下面做inverse
    dX_2 = dX.T * dX
    # I-dX * (dX'*dX)^-1 * dX'
    #python無法計算0矩陣的inverse，用判斷式處理
    if  np.sum(dX_2) == 0:
        M = np.identity(NumObs-lag_p-1) - dX * dX.T
    else:
        M = np.identity(NumObs-lag_p-1) - dX * dX_2.I * dX.T
    
    R0, R1 = dY.T * M, Ys.T * M
    
    S00 = R0 * R0.T / (NumObs-lag_p-1)
    S01 = R0 * R1.T / (NumObs-lag_p-1)
    S10 = R1 * R0.T / (NumObs-lag_p-1)
    S11 = R1 * R1.T / (NumObs-lag_p-1)
    
    #計算廣義特徵值與廣義特徵向量
    eigValue_lambda, eigvecs = eigh(S10 * S00.I * S01, S11, eigvals_only=False)
    
    # 排序特徵向量Eig_vector與特徵值lambda
    sort_ind = np.argsort(-eigValue_lambda)
    #eigValue_lambda = eigValue_lambda[sort_ind]
   
    eigVecs = eigvecs[:, sort_ind]
    #將所有eigenvector同除第一行的總和
    eigVecs_st = eigVecs/np.sum(np.absolute(eigVecs[:,0][0:2])) 
   
    #eigValue_lambda = eigValue_lambda.reshape( len(eigValue_lambda) , 1)
    
    #Beta
    jci_beta = eigVecs_st[:,0][0:2].reshape(NumDim,1)

    #Alpha
    a = np.mat(eigVecs_st[:,0])
    jci_a = S01 * a.T
    jci_alpha = jci_a/np.sum(np.absolute(jci_a)) 
    
    #初始化 c0, d0, c1, d1
    c0 , d0 = 0, 0
    c1 , d1 = np.zeros([NumDim, 1]), np.zeros([NumDim, 1])
    '''
    #標準化eigvecs以利計算co,do
    if len(eigVecs_s) ==3:
        for i in range(0,len(eigVecs_s)):
            a1 = eigVecs_s[0,i]
            a2 = eigVecs_s[1,i]
            a3 = eigVecs_s[2,i]
            total = abs(a1)+abs(a2)+abs(a3)
            eigVecs_s[0,i] = a1/total
            eigVecs_s[1,i] = a2/total
            eigVecs_s[2,i] = a3/total
    else:
         for i in range(0,len(eigVecs_s)):
            a1 = eigVecs_s[0,i]
            a2 = eigVecs_s[1,i]
            total = abs(a1)+abs(a2)
            eigVecs_s[0,i] = a1/total
            eigVecs_s[1,i] = a2/total
    '''
    #計算 c0, d0, c1, d1，與殘差及VEC項的前置
    if model_type == 1:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        
    elif model_type == 2:
        c0 = eigVecs_st[-1, 0:1]
        W = dY - (Ys[:,0:2] * jci_beta + numpy.matlib.repmat(c0, NumObs-lag_p-1, 1) )* jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
    
    elif model_type == 3:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W
        P = P.T
        c = P[:,-1]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
    
    elif model_type == 4:
        d0 = eigVecs_st[-1, 0:1]
        W = dY - (Ys[:,0:2] * jci_beta + np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) * d0) * jci_alpha.T
        P = dX.I * W
        P = P.T
        c = P[:,-1]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        
    elif model_type == 5:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        c = P[:,-2]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        d = P[:,-1]
        d0 = jci_alpha.I * d
        d1 = d - jci_alpha * d0
    #計算殘差    
    ut = W - dX * P.T
    Ct_all = jci_alpha*c0 + c1 + jci_alpha*d0 +d1

    #計算VEC項
    gamma = []
    for bi in range(1,lag_p+1):
        Bq = P[:, (bi-1)*NumDim : bi * NumDim]
        gamma.append(Bq)
    temp1 = np.dot(np.dot(jci_beta.transpose(),S11[0:2,0:2]),jci_beta)
    omega_hat = S00[0:2,0:2] - np.dot(np.dot(jci_alpha,temp1),jci_alpha.transpose())
    #把Ct統整在一起
    Ct=[]
    Ct.append(c0)
    Ct.append(d0)
    Ct.append(c1)
    Ct.append(d1)
    
    return jci_alpha, jci_beta, Ct, ut, gamma, Ct_all, omega_hat



def JCItest_withTrace(X_data,model_type,lag_p):
    #trace test
    if model_type == 'model1':
        model_type = 1
    elif model_type == 'model2':
        model_type = 2
    elif model_type == 'model3':
        model_type = 3
    [NumObs,NumDim] = X_data.shape

    dY_ALL = X_data[1:, :] - X_data[0:-1, :] 
    dY = dY_ALL[lag_p:, :] #DY
    Ys = X_data[lag_p:-1, :] #Lag_Y
    
    #底下開始處理估計前的截距項與時間趨勢項
    if lag_p == 0:
        if model_type == 1:
            dX = np.zeros([NumObs-1, NumDim]) #DLag_Y
        elif model_type == 2:
            dX = np.zeros([NumObs-1, NumDim]) #DLag_Y
            Ys = np.hstack( ( Ys, np.ones((NumObs-lag_p-1,1)) ) )#Lag_Y
        elif model_type == 3:
            dX = np.ones((NumObs-lag_p-1,1)) #DLag_Y
        elif model_type == 4:
            dX = np.ones((NumObs-lag_p-1,1)) #DLag_Y
            Ys = np.hstack( ( Ys, np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )#Lag_Y
        elif model_type == 5:
            dX = np.hstack( ( np.ones((NumObs-lag_p-1,1)) , np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )
    
    elif lag_p>0:
        dX = np.zeros([NumObs-lag_p-1, NumDim * lag_p]) #DLag_Y
        for xi in range(lag_p):
            dX[:, xi * NumDim:(xi + 1) * NumDim] = dY_ALL[lag_p - xi -1 :NumObs - xi - 2, :]
        if model_type == 2:
            Ys = np.hstack( ( Ys, np.ones((NumObs-lag_p-1,1)) ) )
        elif model_type == 3:
            dX = np.hstack( ( dX, np.ones((NumObs-lag_p-1,1)) ) )
        elif model_type == 4:
            Ys = np.hstack( ( Ys, np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )
            dX = np.hstack( ( dX, np.ones((NumObs-lag_p-1,1)) ) )
        elif model_type == 5:
            dX = np.hstack( ( dX, np.ones((NumObs-lag_p-1,1)) , np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) ) )
    
    # 準備開始估計，先轉成matrix，計算比較直觀
    dX, dY, Ys = np.mat(dX), np.mat(dY), np.mat(Ys)

    # 先求dX'*dX 方便下面做inverse
    dX_2 = dX.T * dX
    # I-dX * (dX'*dX)^-1 * dX'
    #python無法計算0矩陣的inverse，用判斷式處理
    if  np.sum(dX_2) == 0:
        M = np.identity(NumObs-lag_p-1) - dX * dX.T
    else:
        M = np.identity(NumObs-lag_p-1) - dX * dX_2.I * dX.T
    
    R0, R1 = dY.T * M, Ys.T * M
    
    S00 = R0 * R0.T / (NumObs-lag_p-1)
    S01 = R0 * R1.T / (NumObs-lag_p-1)
    S10 = R1 * R0.T / (NumObs-lag_p-1)
    S11 = R1 * R1.T / (NumObs-lag_p-1)
    
    #計算廣義特徵值與廣義特徵向量
    eigValue_lambda, eigvecs = eigh(S10 * S00.I * S01, S11, eigvals_only=False)
    
    # 排序特徵向量Eig_vector與特徵值lambda
    sort_ind = np.argsort(-eigValue_lambda)
    eigValue_lambda = eigValue_lambda[sort_ind]
   
    eigVecs = eigvecs[:, sort_ind]
    #將所有eigenvector同除第一行的總和
    #eigVecs_st = eigVecs/np.sum(np.absolute(eigVecs[:,0][0:2])) 
   
    eigValue_lambda = eigValue_lambda.reshape( len(eigValue_lambda) , 1)
    
    #Beta
    #jci_beta = eigVecs_st[:,0][0:2].reshape(NumDim,1)
    jci_beta = eigVecs[:,0][0:2].reshape(NumDim,1)
    
    '''
    #Alpha
    a = np.mat(eigVecs_st[:,0])
    jci_a = S01 * a.T
    jci_alpha = jci_a/np.sum(np.absolute(jci_a)) 
    '''
    #Alpha
    a = np.mat(eigVecs[:,0])
    jci_alpha = S01 * a.T
    
    #初始化 c0, d0, c1, d1
    c0 , d0 = 0, 0
    c1 , d1 = np.zeros([NumDim, 1]), np.zeros([NumDim, 1])

    #計算 c0, d0, c1, d1，與殘差及VEC項的前置
    if model_type == 1:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        cvalue = [12.3329, 4.1475]
    elif model_type == 2:
        #c0 = eigVecs_st[-1, 0:1]
        c0 = eigVecs[-1, 0:1]
        W = dY - (Ys[:,0:2] * jci_beta + numpy.matlib.repmat(c0, NumObs-lag_p-1, 1) )* jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        cvalue = [20.3032, 9.1465]
    elif model_type == 3:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W
        P = P.T
        c = P[:,-1]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        cvalue = [15.4904, 3.8509]
    elif model_type == 4:
        #d0 = eigVecs_st[-1, 0:1]
        d0 = eigVecs[-1, 0:1]
        W = dY - (Ys[:,0:2] * jci_beta + np.arange(1,NumObs-lag_p,1).reshape(NumObs-lag_p-1,1) * d0) * jci_alpha.T
        P = dX.I * W
        P = P.T
        c = P[:,-1]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        cvalue = [25.8863, 12.5142]
    elif model_type == 5:
        W = dY - Ys * jci_beta * jci_alpha.T
        P = dX.I * W  # [B1,...,Bq]
        P = P.T
        c = P[:,-2]
        c0 = jci_alpha.I * c
        c1 = c - jci_alpha * c0
        d = P[:,-1]
        d0 = jci_alpha.I * d
        d1 = d - jci_alpha * d0
        cvalue = [18.3837, 3.8395]
    #計算殘差    
    ut = W - dX * P.T
    Ct_all = jci_alpha*c0 + c1 + jci_alpha*d0 +d1

    #計算VEC項
    gamma = []
    for bi in range(1,lag_p+1):
        Bq = P[:, (bi-1)*NumDim : bi * NumDim]
        gamma.append(Bq)
    temp1 = np.dot(np.dot(jci_beta.transpose(),S11[0:2,0:2]),jci_beta)
    omega_hat = S00[0:2,0:2] - np.dot(np.dot(jci_alpha,temp1),jci_alpha.transpose())
    #把Ct統整在一起
    Ct=[]
    Ct.append(c0)
    Ct.append(d0)
    Ct.append(c1)
    Ct.append(d1)
    
    TraceTest_H = []
    TraceTest_T = []
    for rn in range(0,NumDim):
        eig_lambda = np.cumprod(1-eigValue_lambda[rn:NumDim,:])
        trace_stat = -2 * np.log(eig_lambda[-1] ** ((NumObs-lag_p-1)/2))
        TraceTest_H.append(cvalue[rn] < trace_stat)
        TraceTest_T.append(trace_stat)
    return TraceTest_H, TraceTest_T, jci_alpha, jci_beta, Ct, ut, gamma, Ct_all, omega_hat

def JCI_select_model3(Row_Y,opt_q):  
    #
    #如果model3 rank1不拒絕，則為model3，其餘不做
    [NumObs, k] = Row_Y.shape
    opt_p = opt_q + 1
    Tl = NumObs - opt_p
    
    TraceTest_table = np.zeros([5, k])
    for mr in range(0,5):
        tr_H, _, _, _, _, ut, _, _, _ = JCItest_withTrace(Row_Y, mr+1, opt_q)
        #把結果存起來，True是拒絕，False是不拒絕，tr_H[0]是Rank0,tr_H[1]是Rank1
        TraceTest_table[mr,:] = tr_H
    #挑出被選的Rank1模型
    if TraceTest_table[2,0] == 1 and TraceTest_table[2,1] == 0 :
        return 3
        #拒絕R0，不拒絕R1，該模型的最適Rank為R1，並把該模型與Rank1的BIC值存起來
    else :
        return 0 

def JCI_AutoSelection(Row_Y,opt_q):
    #論文中的BIC model selection
    [NumObs, k] = Row_Y.shape
    opt_p = opt_q + 1
    Tl = NumObs - opt_p
    
    TraceTest_table = np.zeros([5, k])
    BIC_table = np.zeros([5, 1])
    BIC_List = np.ones([5, 1]) * np.Inf
    opt_model_num = 0
    for mr in range(0,5):
        tr_H, _, _, _, _, ut, _, _, _ = JCItest_withTrace(Row_Y, mr+1, opt_q)
        #把結果存起來，True是拒絕，False是不拒絕，tr_H[0]是Rank0,tr_H[1]是Rank1
        TraceTest_table[mr,:] = tr_H
        #以下計算BIC，僅計算Rank1
        eps = np.mat(ut)
        sq_Res_r1 = eps.T * eps / Tl
        errorRes_r1 = eps * sq_Res_r1.I * eps.T
        trRes_r1 = np.trace(errorRes_r1)
        L = (-k*Tl*0.5)*np.log(2*np.pi) - (Tl*0.5)*np.log(np.linalg.det(sq_Res_r1)) -0.5*trRes_r1
        
        if mr==0:
            #alpha(k,1) + beta(k,1) + q*Gamma(k,k)
            deg_Fred = 2*k + opt_q*(k*k)
        elif mr==1:
            #alpha(k,1) + beta(k,1) + C0(1,1) + q*Gamma(k,k)
            deg_Fred = 2*k + 1 + opt_q*(k*k)
        elif mr==2:
            #alpha(k,1) + beta(k,1) + C0(1,1) + C1(k,1) + q*Gamma(k,k)
            deg_Fred = 3*k + 1 + opt_q*(k*k)
        elif mr==3:
            #alpha(k,1) + beta(k,1) + C0(1,1) + D0(1,1) + C1(k,1) + q*Gamma(k,k)
            deg_Fred = 3*k + 2 + opt_q*(k*k)
        elif mr==4:
            #alpha(k,1) + beta(k,1) + C0(1,1) + D0(1,1) + C1(k,1) + D1(k,1) + q*Gamma(k,k)
            deg_Fred = 4*k + 2 + opt_q*(k*k)
        #把Rank1各模型的BIC存起來
        BIC_table[mr] = -2*np.log(L) + deg_Fred*np.log(NumObs*k)
        
        #挑出被選的Rank1模型
        if TraceTest_table[mr,0] == 1 and TraceTest_table[mr,1] == 0 :
            #拒絕R0，不拒絕R1，該模型的最適Rank為R1，並把該模型與Rank1的BIC值存起來
            BIC_List[mr] = BIC_table[mr]
            opt_model_num += 1
        elif TraceTest_table[mr,0] == 0 and TraceTest_table[mr,1] == 0:
            #不拒絕R0，那R1應該是不用測，該模型的最適Rank為R0，紀錄為NaN
            continue
        elif TraceTest_table[mr,0] == 0 and TraceTest_table[mr,1] == 1:
            #不拒絕R0，那R1應該是不用測，該模型的最適Rank為R0，紀錄為NaN
            continue
        elif TraceTest_table[mr,0] == 1 and TraceTest_table[mr,1] == 1:
            #拒絕R0且拒絕R1，該模型的最適Rank為R2，紀錄為NaN
            continue
    
    BIC_List = BIC_List.tolist()
    #找出有紀錄的BIC中最小值，即為Opt_model，且Opt_model+1就對應我們的模型編號
    Opt_model = BIC_List.index(min(BIC_List))
    '''
    #分model1~3/4~5討論
    #model1~3
    if BIC_List[0] != [float('Infinity')] : 
        Opt_model_no_trend = 0
    if BIC_List[1] != [float('Infinity')] : 
        Opt_model_no_trend = 1
    if BIC_List[2] != [float('Infinity')]:
        Opt_model_no_trend = 2
    #model4~5
    Opt_model_trend = 0    
    if BIC_List[3] != [float('Infinity')] :
        Opt_model_trend = 3
    if BIC_List[4] != [float('Infinity')] :
        Opt_model_trend = 4
    if (Opt_model_trend  == 3 or Opt_model_trend  == 4):
        Opt_model = Opt_model_trend
    else:
        Opt_model =  Opt_model_no_trend
    '''
    if opt_model_num == 0:
        #如果opt_model_num是0，代表沒有最適模型或最適模型為Rank0
        return  0
    else:
        #如果opt_model_num不是0，則Opt_model+1模型的Rank1即為我們最適模型
        return Opt_model+1
        
'''
def Companion_para_D(mod_type, lag_p, mod_para, mod_gamma):
    mod_para=[]
    mod_para.append(F_a)
    mod_para.append(F_b)
    mod_para.extend(F_ct)
    
    lag_p = lag_q + 1
    alpha = mod_para[0]
    beta = mod_para[1]
    if lag_q > 0:
        #建立～A
        tilde_A_11 = alpha
        tilde_A_21 = np.zeros([NumDim*lag_q, 1])
        tilde_A_12 = np.zeros([NumDim, NumDim*lag_q])
        
        #建立～B
        tilde_B_11 = beta
        #tilde_A_21與tilde_B_21為相同維度的0矩陣，不重複建立變數
        tilde_B_3 = np.zeros([NumDim + NumDim*lag_q, NumDim*lag_q])
        
        #用同一個迴圈同時處理～A與～B
        for qi in range(lag_q):
            tilde_A_12[0:NumDim,qi*NumDim:(qi+1)*NumDim] = F_gam[qi]
            tilde_B_3[qi*NumDim:NumDim*(2+qi),qi*NumDim:(qi+1)*NumDim] = np.vstack([ np.eye(NumDim), -np.eye(NumDim)])
        tilde_A_22 = np.eye(NumDim*lag_q)
        tilde_A = np.hstack([ np.vstack([tilde_A_11,tilde_A_21]),  np.vstack([tilde_A_12,tilde_A_22 ])])
        tilde_B = np.hstack([ np.vstack([tilde_B_11,tilde_A_21]), tilde_B_3])
    
    elif lag_q == 0:
        tilde_A = alpha
        tilde_B = beta
    
    #建立初始～Ct
    tilde_Ct = np.zeros([NumDim*(1+lag_q), 1])
    
    #建立初始～J
    tilde_J = np.zeros([1, 1+NumDim*(lag_q)])
    tilde_J[0,0] = 1
    
    #建立初始～Omega
    tilde_Sigma = np.zeros([NumDim*(1+lag_q), NumDim*(1+lag_q)])
    
    if model_type == 2:
        tilde_Ct[0:NumDim,] = np.dot(alpha,mod_para[2])
    elif  model_type == 3:
        tilde_Ct[0:NumDim,] = np.dot(alpha,mod_para[2]) + mod_para[4]
    elif  model_type == 4:
        raise ValueError("Error model_type can't be 4 or 5")
    elif  model_type == 5:
        raise ValueError("Error model_type can't be 4 or 5")
    
    inv_BA = np.linalg.inv(np.dot(-tilde_B.transpose(), tilde_A))
    JBA = np.dot(np.dot(tilde_J , inv_BA),tilde_B.transpose())
    
    Com_mean = np.dot(JBA, tilde_Ct)
    Com_slope = 0
    Omega = np.zeros([NumDim, NumDim])
    for ui in range(len(F_ut)):
        Omega = Omega + np.dot(F_ut[ui,:].transpose(),F_ut[ui,:])
    tilde_Sigma[0:NumDim, 0:NumDim] = Omega/ (len(F_ut)-1)
    Com_var = np.dot(np.dot(JBA, tilde_Sigma), JBA.transpose())
    '''
    
def Companion_para_H(mod_type, lag_p, mod_para, mod_gamma, ut):
    NumDim = 2
    alpha = mod_para[0]
    beta = mod_para[1]
    if lag_p > 0:
        #建立～A
        tilde_A_11 = alpha
        tilde_A_21 = np.zeros([NumDim*lag_p, 1])
        tilde_A_12 = np.zeros([NumDim, NumDim*lag_p])
        
        #建立～B
        tilde_B_11 = beta
        #tilde_A_21與tilde_B_21為相同維度的0矩陣，不重複建立變數
        tilde_B_3 = np.zeros([NumDim + NumDim*lag_p, NumDim*lag_p])
        
        #用同一個迴圈同時處理～A與～B
        for qi in range(lag_p):
            tilde_A_12[0:NumDim,qi*NumDim:(qi+1)*NumDim] = mod_gamma[qi]
            tilde_B_3[qi*NumDim:NumDim*(2+qi),qi*NumDim:(qi+1)*NumDim] = np.vstack([ np.eye(NumDim), -np.eye(NumDim)])
        tilde_A_22 = np.eye(NumDim*lag_p)
        tilde_A = np.hstack([ np.vstack([tilde_A_11,tilde_A_21]),  np.vstack([tilde_A_12,tilde_A_22 ])])
        tilde_B = np.hstack([ np.vstack([tilde_B_11,tilde_A_21]), tilde_B_3])
    
    elif lag_p == 0:
        tilde_A = alpha
        tilde_B = beta
    
    #建立初始～C0
    tilde_C0 = np.zeros([NumDim*(1+lag_p), 1])
    
    #建立初始～Ct
    tilde_Ct = np.zeros([NumDim*(1+lag_p), 1])
    
    #建立初始～J
    tilde_J = np.zeros([1, 1+NumDim*(lag_p)])
    tilde_J[0,0] = 1
    
    #建立初始～Omega
    tilde_Sigma = np.zeros([NumDim*(1+lag_p), NumDim*(1+lag_p)])
    
    if mod_type == 2:
        tilde_C0[0:NumDim,] = np.dot(alpha,mod_para[2].reshape(1,1))
    elif  mod_type == 3:
        tilde_C0[0:NumDim,] = np.dot(alpha,mod_para[2].reshape(1,1)) + mod_para[4]
    elif  mod_type == 4:
        C0 = mod_para[2]
        C1 = mod_para[4]
        D0 = mod_para[3]
        tilde_C0[0:NumDim,] = np.dot(alpha,C0) + C1 + np.dot(alpha,D0).reshape(NumDim,1)
        tilde_Ct[0:NumDim,] = np.dot(alpha,D0).reshape(NumDim,1)
    elif  mod_type == 5:
        C0 = mod_para[2]
        C1 = mod_para[4]
        D0 = mod_para[3]
        D1 = mod_para[5]
        tilde_C0[0:NumDim,] = np.dot(alpha,C0) + C1 + np.dot(alpha,D0) + D1
        tilde_Ct[0:NumDim,] = np.dot(alpha,D0) + D1
    
    inv_BA = np.linalg.inv(np.dot(-tilde_B.transpose(), tilde_A))
    JBA = np.dot(np.dot(tilde_J , inv_BA),tilde_B.transpose())
    
    Com_intcpt = np.dot(JBA, tilde_C0)
    Com_slope = np.dot(JBA, tilde_Ct)
    
    tilde_Sigma[0:NumDim, 0:NumDim] = np.dot(ut.transpose(),ut) / (len(ut)-1)
    #tilde_2AB = 2*np.eye(1 + NumDim*lag_p) + np.dot(tilde_B.transpose(), tilde_A)
    #tilde_J2AB = np.dot(np.dot( tilde_J ,np.linalg.inv(tilde_2AB)) , tilde_B.transpose())
    
    #Com_var = np.dot(np.dot(JBA, tilde_Sigma), tilde_J2AB.transpose())
    Com_var = np.dot(np.dot(JBA, tilde_Sigma), JBA.transpose())
    return Com_intcpt, Com_slope, Com_var

def Companion_para_J(mod_type, lag_p, mod_para, mod_gamma, ut):

    NumDim = 2
    alpha = mod_para[0]
    beta = mod_para[1]
    if lag_p > 0:
        #建立～A
        tilde_A_11 = alpha
        tilde_A_21 = np.zeros([NumDim*lag_p, 1])
        tilde_A_12 = np.zeros([NumDim, NumDim*lag_p])
        
        #建立～B
        tilde_B_11 = beta
        #tilde_A_21與tilde_B_21為相同維度的0矩陣，不重複建立變數
        tilde_B_3 = np.zeros([NumDim + NumDim*lag_p, NumDim*lag_p])
        
        #用同一個迴圈同時處理～A與～B
        for qi in range(lag_p):
            tilde_A_12[0:NumDim,qi*NumDim:(qi+1)*NumDim] = mod_gamma[qi]
            tilde_B_3[qi*NumDim:NumDim*(2+qi),qi*NumDim:(qi+1)*NumDim] = np.vstack([ np.eye(NumDim), -np.eye(NumDim)])
        tilde_A_22 = np.eye(NumDim*lag_p)
        tilde_A = np.hstack([ np.vstack([tilde_A_11,tilde_A_21]),  np.vstack([tilde_A_12,tilde_A_22 ])])
        tilde_B = np.hstack([ np.vstack([tilde_B_11,tilde_A_21]), tilde_B_3])
        tilde_I = np.eye(1+NumDim*lag_p)
        
    elif lag_p == 0:
        tilde_A = alpha
        tilde_B = beta
        tilde_I = 0
    #建立初始～C0
    tilde_C0 = np.zeros([NumDim*(1+lag_p), 1])
    
    #建立初始～Ct
    tilde_Ct = np.zeros([NumDim*(1+lag_p), 1])
    
    #建立初始～J
    tilde_J = np.zeros([1, 1+NumDim*(lag_p)])
    tilde_J[0,0] = 1
    
    #建立初始～Omega
    tilde_Sigma = np.zeros([NumDim*(1+lag_p), NumDim*(1+lag_p)])
    
    if mod_type == 2:
        tilde_C0[0:NumDim,] = np.dot(alpha,mod_para[2].reshape(1,1))
    elif  mod_type == 3:
        tilde_C0[0:NumDim,] = np.dot(alpha,mod_para[2].reshape(1,1)) + mod_para[4]
    elif  mod_type == 4:
        C0 = mod_para[2]
        C1 = mod_para[4]
        D0 = mod_para[3]
        tilde_C0[0:NumDim,] = np.dot(alpha,C0) + C1 + np.dot(alpha,D0).reshape(NumDim,1)
        tilde_Ct[0:NumDim,] = np.dot(alpha,D0).reshape(NumDim,1)
    elif  mod_type == 5:
        C0 = mod_para[2]
        C1 = mod_para[4]
        D0 = mod_para[3]
        D1 = mod_para[5]
        tilde_C0[0:NumDim,] = np.dot(alpha,C0) + C1 + np.dot(alpha,D0) + D1
        tilde_Ct[0:NumDim,] = np.dot(alpha,D0) + D1
    
    '''
    inv_BA = np.linalg.inv(np.dot(-tilde_B.transpose(), tilde_A))
    JBA = np.dot(np.dot(tilde_J , inv_BA),tilde_B.transpose())
    
    Com_intcpt = np.dot(JBA, tilde_C0)
    Com_slope = np.dot(JBA, tilde_Ct)
    '''
    tilde_Sigma[0:NumDim, 0:NumDim] = np.dot(ut.transpose(),ut)/(len(ut)-1)
    if lag_p == 0:
        inv_JBA =np.dot(tilde_J, np.linalg.inv(tilde_I - np.dot(-tilde_B.transpose(), tilde_A) ))
        tilde_JBAB = np.dot( inv_JBA,tilde_B.transpose())
        
        Com_slope = np.dot(tilde_JBAB , tilde_Ct)
        Com_intcpt = np.dot(tilde_JBAB , tilde_C0)
        Com_var = np.dot(np.dot(tilde_JBAB , tilde_Sigma) , tilde_JBAB.transpose())
        
    elif lag_p > 0:
        inv_BA =  np.linalg.inv(np.dot(-tilde_B.transpose(), tilde_A)- tilde_I )
        tilde_JBAB = np.dot( np.dot(tilde_J , inv_BA) ,tilde_B.transpose())
        
        Com_slope = np.dot(tilde_JBAB , tilde_Ct)
        Com_intcpt = np.dot(tilde_JBAB , tilde_C0)
        Com_var =np.dot(np.dot(tilde_JBAB , tilde_Sigma), tilde_JBAB.transpose())
        
    return Com_intcpt, Com_slope, Com_var

def Johansen_mean(alpha,beta,gamma,mu,lagp,NumDim=2): 
    #論文中的closed form mean
    #lagp指的是VECM的LAG期數
    sumgamma = np.zeros([NumDim, NumDim])
    for i in range(0,lagp):
        sumgamma =sumgamma+gamma[i]
    GAMMA = np.eye(NumDim) - sumgamma 
    #計算正交化的alpha,beta
    alpha_orthogonal = alpha.copy()  
    alpha_t = alpha.transpose()
    alpha_orthogonal[1,0] = (-(alpha_t[0,0]*alpha_orthogonal[0,0])) / alpha_t[0,1]    
    alpha_orthogonal = alpha_orthogonal/sum(abs(alpha_orthogonal))
    beta_orthogonal = beta.copy()  
    beta_t = beta.transpose()
    beta_orthogonal[1,0] = -((beta_t[0,0]*beta_orthogonal[0,0])) / beta_t[0,1]    
    beta_orthogonal = beta_orthogonal/sum(abs(beta_orthogonal)) 
    #計算MEAN
    temp1 = np.linalg.inv(np.dot(np.dot(alpha_orthogonal.transpose(), GAMMA),beta_orthogonal))
    C = np.dot(np.dot(beta_orthogonal,temp1),alpha_orthogonal.transpose())
    temp2 = np.linalg.inv(np.dot(alpha.transpose(),alpha))
    alpha_hat = np.dot(alpha,temp2)
    temp3 = np.dot(GAMMA,C) - np.eye(NumDim)
    C0 = np.mat(mu[0])
    C1 = np.mat(mu[2])
    D0 = np.mat(mu[1])
    D1 = np.mat(mu[3])
    C0 = alpha*C0 + C1 + alpha*D0 + D1
    Ct = alpha*D0 + D1
    expect_intcept = np.dot(np.dot(alpha_hat.transpose(),temp3),C0)
    expect_slope = np.dot(np.dot(alpha_hat.transpose(),temp3),Ct)
    return expect_intcept, expect_slope

def Johansen_std(alpha,beta,ut,rank=1):
    temp1 = np.eye(rank)+np.dot(beta.transpose(),alpha)
    temp2 = np.kron(temp1,temp1)
    temp3 = np.linalg.inv(np.eye(rank)-temp2)
    omega = np.dot(ut.transpose(),ut)/(len(ut)-1)
    temp4 = np.dot(np.dot(beta.transpose(),omega),beta)
    var = np.dot(temp3,temp4)
    #std = np.sqrt(var)
    return var
def Johansen_std_correct(alpha,beta,ut,mod_gamma,lag_p,rank=1):
    #論文中的closed form std
    NumDim = 2
    if lag_p > 0:
        #建立～A
        tilde_A_11 = alpha
        tilde_A_21 = np.zeros([NumDim*lag_p, 1])
        tilde_A_12 = np.zeros([NumDim, NumDim*lag_p])
        
        #建立～B
        tilde_B_11 = beta
        #tilde_A_21與tilde_B_21為相同維度的0矩陣，不重複建立變數
        tilde_B_3 = np.zeros([NumDim + NumDim*lag_p, NumDim*lag_p])
        
        #用同一個迴圈同時處理～A與～B
        for qi in range(lag_p):
            tilde_A_12[0:NumDim,qi*NumDim:(qi+1)*NumDim] = mod_gamma[qi]
            tilde_B_3[qi*NumDim:NumDim*(2+qi),qi*NumDim:(qi+1)*NumDim] = np.vstack([ np.eye(NumDim), -np.eye(NumDim)])
        tilde_A_22 = np.eye(NumDim*lag_p)
        tilde_A = np.hstack([ np.vstack([tilde_A_11,tilde_A_21]),  np.vstack([tilde_A_12,tilde_A_22 ])])
        tilde_B = np.hstack([ np.vstack([tilde_B_11,tilde_A_21]), tilde_B_3])
    
    elif lag_p == 0:
        tilde_A = alpha
        tilde_B = beta
    tilde_Sigma = np.zeros([NumDim*(lag_p+1), NumDim*(lag_p+1)])
    tilde_Sigma[0:NumDim, 0:NumDim] = np.dot(ut.transpose(),ut)/(len(ut)-1)
    tilde_J = np.zeros([1, 1+NumDim*(lag_p)])
    tilde_J[0,0] = 1
    if lag_p == 0  :
        temp1 = np.eye(rank)+np.dot(beta.transpose(),alpha)
        temp2 = np.kron(temp1,temp1)
        temp3 = np.linalg.inv(np.eye(rank)-temp2)
        omega = np.dot(ut.transpose(),ut)/(len(ut)-1)
        temp4 = np.dot(np.dot(beta.transpose(),omega),beta)
        var = np.dot(temp3,temp4)
    else:   
        temp1 =np.eye(NumDim*(lag_p+1)-1)+np.dot(tilde_B.transpose(),tilde_A)
        temp2 = np.kron(temp1,temp1)
        k = (NumDim*(lag_p+1)-1)*(NumDim*(lag_p+1)-1)
        temp3 = np.linalg.inv(np.eye(k)-temp2)
        temp4 = np.dot(np.dot(tilde_B.transpose(),tilde_Sigma),tilde_B)
        temp4 = temp4.flatten('F')
        temp5 = np.dot(temp3,temp4)
        sigma_telta_beta = np.zeros([NumDim*(lag_p+1)-1, NumDim*(lag_p+1)-1])
        for i in range(NumDim*(lag_p+1)-1):
            for j in range(NumDim*(lag_p+1)-1):
                sigma_telta_beta[i][j]= temp5[0,i+j*(NumDim*(lag_p+1)-1)]
        var = np.dot(np.dot(tilde_J, sigma_telta_beta), tilde_J.transpose())
    return var
def Iteration_Mean_Std(mod_type, lag_p, mod_para, mod_gamma, St, od, Forma_End):
    #迭帶
    [St_length, NumDim] = St.shape
    alpha = np.mat(mod_para[0])
    beta = np.mat(mod_para[1])
    C0 = 0
    C1 = 0
    D0 = 0
    D1 = 0
    C0, C1, D0, D1 = np.mat(C0), np.mat(C1), np.mat(D0), np.mat(D1)
    tf = np.zeros([1,St_length])
    if mod_type == 2:
        C0 = np.mat(mod_para[2])
    elif  mod_type == 3:
        C0 = np.mat(mod_para[2])
        C1 = np.mat(mod_para[4])
    elif  mod_type == 4:
        C0 = np.mat(mod_para[2])
        C1 = np.mat(mod_para[4])
        D0 = np.mat(mod_para[3])
        #tf = np.linspace(-od+1,St_length-od,St_length)
        tf = np.linspace(0,St_length,St_length)
        tf = np.mat(tf)
    elif  mod_type == 5:
        C0 = np.mat(mod_para[2])
        C1 = np.mat(mod_para[4])
        D0 = np.mat(mod_para[3])
        D1 = np.mat(mod_para[5])
        #tf = np.linspace(-od+1,St_length-od,St_length)
        tf = np.linspace(0,St_length,St_length)
        tf = np.mat(tf)
    
    S_hat = np.zeros([St_length, NumDim])
    S_hat[0:od,:] = St[0:od,:]
    S_hat = np.mat(S_hat)
    
    for si in range(od, St_length):
            m1 = S_hat[si-1,:] * beta * alpha.T + C0.T * alpha.T + C1.T 
            m2 = (D0 * alpha.T) * tf[0,si]
            m3 = D1.T * tf[0,si]
            
            dXt = np.matrix([0,0])
            for bi in range(lag_p):
                dXt = dXt + (S_hat[si-bi-1,:] - S_hat[si-bi-2,:] ) * mod_gamma[bi]
            
            S_hat[si,:] = S_hat[si-1,:] + m1 + m2 + m3 + dXt
    
    Ir_Mean = S_hat * beta
    spread = St * beta
    '''
     # 將均值兩次差分
    Mean_Trend_d2 = np.zeros([St.shape[0],1])
    Mean_Trend_d2[2:] = np.diff(Ir_Mean,n=2,axis=0)
    Mean_Trend_d2[0:2] = Mean_Trend_d2[2]
    smsp = np.full([1,1], np.nan)
    # 兩次差分後夠接近0的就視為收斂
    for si in range(St.shape[0]):
        if abs(Mean_Trend_d2[si]) < 0.000001:
            smsp = si
            break
    #沒有收斂點 bang掉 
    if np.isnan(smsp):
        Ir_Mean = np.zeros([St.shape[0],1])
        Ir_std = 0
        smsp = St.shape[0]
    #尾盤不收斂 bang掉
    elif Mean_Trend_d2[St.shape[0]-1,0] > 0.000001:
        Ir_Mean = np.zeros([St.shape[0],1])
        Ir_std = 0
        smsp = St.shape[0]
    else:
    
        StMean = Ir_Mean
    
        #ErrorT = Ir_Mean[smsp: ,]- spread[smsp: ,]
        #St_Std = np.sqrt(ErrorT.T * ErrorT / len(Mean_Trend[smsp: ,]))
        I_mer = np.sum(np.square(Ir_Mean[od:Forma_End+1] - spread[od:Forma_End+1]))
        Ir_std = np.sqrt(I_mer / (Forma_End-od+1) )
    '''
    #I_mer = np.sum(np.square(Ir_Mean[od:Forma_End+1] - spread[od:Forma_End+1]))
    #Ir_std = np.sqrt(I_mer / (Forma_End-od+1) )
    
    return Ir_Mean#, Ir_std #, smsp
