# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:27:14 2020

@author: Hua
"""
import numpy as np
import pandas as pd
from module.Matrix_function import order_select
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import module.JCItestpara_20201113 as jci


def get_key(dict, value):
    tmp = [k for k, v in dict.items() if v == value]
    # print(tmp[0])
    return tmp[0]


def formation_table(Smin, inNum):
    col_name = {name: i for i, name in enumerate(Smin.columns.values)}
    # print(col_name)
    LSmin = np.log(Smin)
    # LSmin = Smin
    maxcompanynu = Smin.shape[1]  # 找出有多少檔
    ind = mt.Binal_comb(range(maxcompanynu))
    ind = np.hstack((ind, np.zeros([ind.shape[0], 7])))
    # ind.columns = [0:S1_inx,1:S2_inx,2:opt_q, 3:Johansen intercept, 4:Johansen slope, 5:std,6:Model,7:W1,8:W2]
    DailyNum = len(Smin)
    # cy為Naturn Log共整合序列，以Capital Weight構成
    cy = np.zeros([DailyNum, ind.shape[0]])
    # cy_mean為共整合序列均值，以Capital Weight構成
    cy_mean = np.zeros([DailyNum, ind.shape[0]])
    B = np.zeros([2, ind.shape[0]])  # B為共整合係數
    CapitW = np.zeros([2, ind.shape[0]])  # CW為資金權重Capital Weight
    # IntegerB = np.zeros([2,ind.shape[0]]) #IB為CW整數化後的共整合係數
    count = 0
    # start_time = time.time()
    for mi in range(ind.shape[0]):
        # for mi in range(1):
        rowS = LSmin.iloc[0:, [int(ind[mi, 0]), int(ind[mi, 1])]]  # 1

        # stock1 = Smin.iloc[inNum-1,[int(ind[mi,0])]]
        # stock2 = Smin.iloc[inNum-1,[int(ind[mi,1])]]
        # ind[mi,0:2] = rowS.columns.values

        ind[mi, 0], ind[mi, 1] = int(col_name[rowS.columns[0]]), int(
            col_name[rowS.columns[1]]
        )
        # print(ind)
        rowAS = np.array(rowS)
        # print("rowAS :",rowAS)
        # 配適 VAR(P) 模型 ，並利用BIC選擇落後期數，max_p意味著會檢查2~max_p
        try:
            max_p = 5
            p = order_select(rowAS, max_p)
            # ADF TEST
            if p < 1:

                continue
            # adf test
            # portmanteau test
            model = VAR(rowAS)

            if model.fit(p).test_whiteness(nlags=5).pvalue < 0.05:

                continue

            # Normality test
            """
            if model.fit(p).test_normality().pvalue < 0.05:

                continue
            """
            opt_model = jci.JCI_AutoSelection(rowAS, p - 1)  # bic based model selection
            # print("model select :",opt_model)
            count += 1
            # opt_model = jci.JCI_select_model3(rowAS,p-1) #no model selection
            # 如果有共整合，紀錄下Model與opt_q
            ind[mi, 2] = p - 1
            ind[mi, 6] = opt_model
            F_a, F_b, F_ct, F_ut, F_gam, ct, omega_hat = jci.JCItestpara_spilCt(
                rowAS, opt_model, p - 1
            )
            Com_para = []
            Com_para.append(F_a)
            Com_para.append(F_b)
            Com_para.extend(F_ct)
            # print(Com_para)
            # 把  arrary.shape(2,1) 的數字放進 shape(2,) 的Serires
            # 取出共整合係數
            B[:, mi] = pd.DataFrame(F_b).stack()
            # 將共整合係數標準化，此為資金權重Capital Weight
            CapitW[:, mi] = B[:, mi] / np.sum(np.absolute(B[:, mi]))
            ind[mi, 7] = CapitW[0, mi]
            ind[mi, 8] = CapitW[1, mi]
            """
            #將資金權重，依股價轉為張數權重
            S1 = CapitW[0][mi]/float(stock1)
            S2 = CapitW[1][mi]/float(stock2)
            
            #將張數權重，做最簡整數比，要求範圍是最大張數+1
            optXY = mt.simp_frac(S1,S2,MaxV+1)
            
            #如果最簡整數比出現[ (MaxV+1) / 1 ] or [ 1 / (MaxV+1) ] 就剃除
            #張數權重整數化後，絕對值小於5的設1（通過），絕對值大於6的設0（沒通過）
            if abs(optXY[0]) <= MaxV  and abs(optXY[1]) <= MaxV:
                ind[mi,4] = 1
                IntegerB[:,mi] = optXY[:]
                ind[mi,7] = optXY[0]
                ind[mi,8] = optXY[1]
            
           """
            # 計算Spread的時間趨勢均值與標準差 model 1-5
            Johansen_intcept, Johansen_slope = jci.Johansen_mean(
                F_a, F_b, F_gam, F_ct, p - 1
            )
            Johansen_var_correct = jci.Johansen_std_correct(
                F_a, F_b, F_ut, F_gam, p - 1
            )
            Johansen_std = np.sqrt(Johansen_var_correct)
            ind[mi, 3] = Johansen_intcept
            ind[mi, 4] = Johansen_slope
            ind[mi, 5] = Johansen_std
            SStd = Johansen_std
            # print("Johansen_intcept :",Johansen_intcept)
            cy_mean[:, mi] = Johansen_intcept + Johansen_slope * np.linspace(
                0, 249, 250
            )
            # 以資金權重建構Naturn Log共整合序列
            cy[:, mi] = pd.DataFrame(np.mat(rowLS) * np.mat(CapitW[:, mi]).T).stack()
            # 拿共整合序列拿去檢定，ADF單根檢定回傳1代表無單根（定態），0代表有單根（非定態）
            # ind[mi,5] = mt.ADFtest_TR(cy[OpenD-1:inNum,mi], opt_p-1 , 0.05)
            # 如果收斂點在Trading Period，設為0（沒通過、不交易），反之設為1
            # if converg_Point < inNum:
            # ind[mi,10] = converg_Point
            # Spend_time = time.time() - start_time
            """
            #畫個圖確認一下
            print(ind[mi,0:2])
            import matplotlib.pyplot as plt
            plotx = [i for i in range(DailyNum)]
            CL = np.zeros((DailyNum,5))
            CL [:,2] = cy_mean[:,mi]
            CL [:,1] = cy_mean[:,mi]+SStd*os
            CL [:,0] = cy_mean[:,mi]+SStd*cs
            CL [:,3] = cy_mean[:,mi]-SStd*os
            CL [:,4] = cy_mean[:,mi]-SStd*cs
            plt.plot(plotx,cy[:,mi],plotx,CL)
            plt.show()
            """
        except:
            continue
    dd = np.zeros([ind.shape[0], 1])
    test_Model = ind[:, 6] != 0
    dd = test_Model
    # print(ind)
    ind_select = ind[dd, :]  # 排除沒有共整合關係的配對
    ind_select = ind_select.tolist()
    # print(ind_select)
    for index in range(len(ind_select)):
        # print(ind_select[index][0] ,ind_select[index][1] )
        ind_select[index][0], ind_select[index][1] = get_key(
            col_name, ind_select[index][0]
        ), get_key(col_name, ind_select[index][1])
        # print(ind_select[index][0] ,ind_select[index][1] )
    # print(ind_select)
    return ind_select


def refactor_formation_table(Smin, inNum):

    rowAS = np.log(Smin)
    B = np.zeros([2, 1])  # B為共整合係數
    CapitW = np.zeros([2, 1])  # CW為資金權重Capital Weight
    # 配適 VAR(P) 模型 ，並利用BIC選擇落後期數，max_p意味著會檢查2~max_p
    for _ in range(1):
        try:
            max_p = 5
            p = order_select(rowAS, max_p)
            # ADF TEST
            if p < 1:
                return []
                # adf test
                # portmanteau test
            model = VAR(rowAS)

            if model.fit(p).test_whiteness(nlags=5).pvalue < 0.05:
                # print("WWWHHYUUU2")
                return []
                # Normality test
            """
            if model.fit(p).test_normality().pvalue < 0.05:
                # print("WHHHYYYY")
                return []
            """
            opt_model = jci.JCI_AutoSelection(rowAS, p - 1)  # bic based model selection
            # print("model select :", opt_model)
            # opt_model = jci.JCI_select_model3(rowAS,p-1) #no model selection
            # 如果有共整合，紀錄下Model與opt_q
            F_a, F_b, F_ct, F_ut, F_gam, ct, omega_hat = jci.JCItestpara_spilCt(
                rowAS, opt_model, p - 1
            )
            Com_para = []
            Com_para.append(F_a)
            Com_para.append(F_b)
            Com_para.extend(F_ct)
            # print(Com_para)
            # 把  arrary.shape(2,1) 的數字放進 shape(2,) 的Serires
            # 取出共整合係數
            B[:, 0] = pd.DataFrame(F_b).stack()
            # 將共整合係數標準化，此為資金權重Capital Weight
            CapitW[:, 0] = B[:, 0] / np.sum(np.absolute(B[:, 0]))

            # 計算Spread的時間趨勢均值與標準差 model 1-5
            Johansen_intcept, Johansen_slope = jci.Johansen_mean(
                F_a, F_b, F_gam, F_ct, p - 1
            )
            Johansen_var_correct = jci.Johansen_std_correct(
                F_a, F_b, F_ut, F_gam, p - 1
            )
            Johansen_std = np.sqrt(Johansen_var_correct)
            # print("Johansen_intcept :", Johansen_intcept)
            Johansen_intcept = Johansen_intcept[0, 0]
            Johansen_std = Johansen_std[0, 0]
            return [
                Johansen_intcept,
                Johansen_std,
                opt_model,
                CapitW[0, 0],
                CapitW[1, 0],
            ]

        except:
            return []
