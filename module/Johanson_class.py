from statsmodels.tsa.api import VAR 
from scipy.linalg import eigh, eig 
import pandas as pd 
import numpy.matlib
import numpy as np

p_value_list = []
timestep = []
def VAR_model(y, p):
    k = len(y.T)  # 幾檔股票
    n = len(y)  # 資料長度
    xt = np.ones((n - p, (k * p) + 1))
    
    for i in range(n - p):
        a = 1
        for j in range(p):
            a = np.hstack((a, y[i + p - j - 1]))
        a = a.reshape([1, (k * p) + 1])
        xt[i] = a
    print("y shape:", y.shape)
    print("p:", p)

# 刪除前的 xt 和 zt 打印
    print("Before deletion:")
    print("xt shape (initial):", xt.shape if 'xt' in locals() else "xt not defined")
    print("zt shape (initial):", y.shape)
    print('xt')
    #print(xt)
    # print('y')
    # print(y)
    zt = np.delete(y, np.s_[0:p], axis=0)
    zt = np.hstack((np.ones((zt.shape[0], 1)), zt))  # 添加一列常數項
    xt = np.asarray(xt, dtype=float)
    print('var model start')
    xt = np.mat(xt)
    zt = np.mat(zt)
    
    beta = (xt.T * xt).I * xt.T * zt  # 計算VAR的參數

    A = zt - xt * beta  # 計算殘差
    sigma = ((A.T) * A) / (n - p)  # 計算殘差的共變異數矩陣
    print('var model end')
    return [sigma, beta]


# 配適 VAR(P) 模型 ，並利用BIC選擇落後期數--------------------------------------------------------------
def order_select(y, max_p):
    print('order select start')
    k = len(y.T)  # 幾檔股票
    n = len(y)  # 資料長度

    bic = np.zeros((max_p, 1))
    print('order select start2')

    for p in range(1, max_p + 1):
       
        sigma = VAR_model(y, p)[0]
        print(f'order select start3 ')
        print(f' {sigma} ')
            #  {np.linalg.det(sigma)}')
        if np.linalg.matrix_rank(sigma) < sigma.shape[0]:
            print("Sigma is singular at lag", p)
            continue  # 或者返回錯誤，根據需求調整
        print(f'{p} {max_p}')
        #sigma = VAR_model(y, p)[0]
        bic[p - 1] = np.log(np.linalg.det(sigma)) + np.log(n) * p * (k * k) / n
    print('order select start4')
    bic_order = int(np.where(bic == np.min(bic))[0] + 1)  # 因為期數p從1開始，因此需要加1
    print('order select end')
    return bic_order

class Johansen: 
    def __init__(self, Smin, date, timestamp):
        self._data = self._tolog(Smin) 
        self.Johansen_mu = 0 
        self.Johansen_stdev = 0 
        self.Johansen_slope = 0 
        self.optimal_model = -1 
        self.B = np.zeros([2, 1]) 
        self.p = 0 
        self.CapitW = np.zeros([2, 1]) 
        self.list_jci_alpha = [] 
        self.list_jci_beta = [] 
        self.list_ut = [] 
        self.list_gamma = [] 
        self.list_Ct = [] 
        #self.p_value
        self.date = date
        self.timestamp = timestamp
    def _tolog(self, data): 
        #print(np.log(data))
        return np.log(data) 
 
    @staticmethod 
    def _order_select(log_data, max_p=5): 
        return order_select(log_data, max_p) 
 
    def _whitenoise(self, log_data): 
        try :
            self.p = self._order_select(log_data) 
        except :
            return False
        if self.p < 1: 
            return False
        model = VAR(log_data)
        try :
            print('cnt times')
            #print(f'model fit {model.fit(self.p).test_whiteness(nlags=5).pvalue}')
            p_value_list.append(model.fit(self.p).test_whiteness(nlags=5).pvalue)
            timestep.append([self.date, self.timestamp])
            if model.fit(self.p).test_whiteness(nlags=5).pvalue < 0.05: 
                return False 
        except :
            print('here5')
            return False
        return True 
 
    def _JCI_AutoSelection(self, Row_Y, opt_q): 
        [NumObs, k] = Row_Y.shape 
        # print("NumObs :", NumObs) 
        # print("k in JCI_AutoSelection :",k) 
        opt_p = opt_q + 1 
        Tl = NumObs - opt_p 
 
        TraceTest_table = np.zeros([5, k]) 
        # print("TraceTest_table :",TraceTest_table) 
        BIC_table = np.zeros([5, 1]) 
        BIC_List = np.ones([5, 1]) * np.Inf 
        opt_model_num = 0 
        for mr in range(0, 5):  # production is 5 
            tr_H, ut = self.JCItest_withTrace( 
                Row_Y, mr+1, opt_q) 
            # 把結果存起來，True是拒絕，False是不拒絕，tr_H[0]是Rank0,tr_H[1]是Rank1 
            TraceTest_table[mr, :] = tr_H 
            # 以下計算BIC，僅計算Rank1 
            eps = np.mat(ut) 
            sq_Res_r1 = eps.T * eps / Tl 
            errorRes_r1 = eps * sq_Res_r1.I * eps.T 
            trRes_r1 = np.trace(errorRes_r1) 
            L = (-k*Tl*0.5)*np.log(2*np.pi) - (Tl*0.5) * np.log(np.linalg.det(sq_Res_r1)) - 0.5*trRes_r1 
 
            if mr == 0: 
                # alpha(k,1) + beta(k,1) + q*Gamma(k,k) 
                deg_Fred = 2*k + opt_q*(k*k) 
            elif mr == 1: 
                # alpha(k,1) + beta(k,1) + C0(1,1) + q*Gamma(k,k) 
                deg_Fred = 2*k + 1 + opt_q*(k*k) 
            elif mr == 2: 
                # alpha(k,1) + beta(k,1) + C0(1,1) + C1(k,1) + q*Gamma(k,k) 
                deg_Fred = 3*k + 1 + opt_q*(k*k) 
            elif mr == 3: 
                # alpha(k,1) + beta(k,1) + C0(1,1) + D0(1,1) + C1(k,1) + q*Gamma(k,k) 
                deg_Fred = 3*k + 2 + opt_q*(k*k) 
            elif mr == 4: 
                # alpha(k,1) + beta(k,1) + C0(1,1) + D0(1,1) + C1(k,1) + D1(k,1) + q*Gamma(k,k) 
                deg_Fred = 4*k + 2 + opt_q*(k*k) 
            # 把Rank1各模型的BIC存起來 
            BIC_table[mr] = -2*np.log(L) + deg_Fred*np.log(NumObs*k) 
 
            # 挑出被選的Rank1模型 
            if TraceTest_table[mr, 0] == 1 and TraceTest_table[mr, 1] == 0: 
                # 拒絕R0，不拒絕R1，該模型的最適Rank為R1，並把該模型與Rank1的BIC值存起來 
                BIC_List[mr] = BIC_table[mr] 
                opt_model_num += 1 
 
        BIC_List = BIC_List.tolist() 
        # 找出有紀錄的BIC中最小值，即為Opt_model，且Opt_model+1就對應我們的模型編號 
        Opt_model = BIC_List.index(min(BIC_List)) 
 
        if opt_model_num == 0: 
            # 如果opt_model_num是0，代表沒有最適模型或最適模型為Rank0 
            return 0 
        else: 
            self.optimal_model = Opt_model 
            # 如果opt_model_num不是0，則Opt_model+1模型的Rank1即為我們最適模型 
            return Opt_model+1 
 
    def JCItest_withTrace(self, X_data, model_type, lag_p):  # 2黨數據 , 模型type , lag基數 
        # trace test 
        NumObs, NumDim = X_data.shape[0], X_data.shape[1]  # (120,2) 
        dY_ALL = X_data[1:, :] - X_data[:-1, :]  # 算截距 
        dY = dY_ALL[lag_p:, :]  # DY 
        Ys = X_data[lag_p:-1, :]  # Lag_Y 
 
        # 底下開始處理估計前的截距項與時間趨勢項 
        if lag_p == 0: 
            if model_type == 1: 
                dX = np.zeros([NumObs-1, NumDim])  #
                # print("model type 1 dX :" ,dX) 
            elif model_type == 2: 
                dX = np.zeros([NumObs-1, NumDim])  # DLag_Y 
                Ys = np.hstack((Ys, np.ones((NumObs-lag_p-1, 1))))  # Lag_Y 
                # print("model type 2 Ys :" ,Ys) 
                # print("model type 2 dX :" ,dX) 
            elif model_type == 3: 
                dX = np.ones((NumObs-lag_p-1, 1))  # DLag_Y 
                # print("model type 3 dX :" ,dX) 
            elif model_type == 4: 
                dX = np.ones((NumObs-lag_p-1, 1))  # DLag_Y 
                Ys = np.hstack( 
                    (Ys, np.arange(1, NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1)))  # Lag_Y 
                # print("model type 4 Ys :" ,Ys) 
                # print("model type 4 dX :" ,dX) 
            elif model_type == 5: 
                dX = np.hstack((np.ones((NumObs-lag_p-1, 1)), np.arange(1, 
                                                                        NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1))) 
                # print("model type 5 dX :" ,dX) 
        elif lag_p > 0: 
            dX = np.zeros([NumObs-lag_p-1, NumDim * lag_p])  # DLag_Y 
            for xi in range(lag_p): 
                dX[:, xi * NumDim:(xi + 1) * NumDim] = dY_ALL[lag_p - 
                                                              xi - 1:NumObs - xi - 2, :] 
            if model_type == 2: 
                Ys = np.hstack((Ys, np.ones((NumObs-lag_p-1, 1)))) 
            elif model_type == 3: 
                dX = np.hstack((dX, np.ones((NumObs-lag_p-1, 1)))) 
                # print("dX lagp",dX) 
            elif model_type == 4: 
                Ys = np.hstack( 
                    (Ys, np.arange(1, NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1))) 
                dX = np.hstack((dX, np.ones((NumObs-lag_p-1, 1)))) 
            elif model_type == 5: 
                dX = np.hstack((dX, np.ones((NumObs-lag_p-1, 1)), 
                                np.arange(1, NumObs-lag_p, 1).reshape(NumObs-lag_p-1, 1))) 
 
        # 準備開始估計，先轉成matrix，計算比較直觀 
        dX, dY, Ys = np.mat(dX), np.mat(dY), np.mat(Ys) 
        # 先求dX'*dX 方便下面做inverse 
        dX_2 = dX.T * dX 
        # I-dX * (dX'*dX)^-1 * dX' 
        # python無法計算0矩陣的inverse，用判斷式處理 
        if np.sum(dX_2) == 0: 
            M = np.identity(NumObs-lag_p-1) - dX * dX.T 
        else: 
            M = np.identity(NumObs-lag_p-1) - dX * dX_2.I * dX.T 
        R0, R1 = dY.T * M, Ys.T * M 
        S00 = R0 * R0.T / (NumObs-lag_p-1) 
        S01 = R0 * R1.T / (NumObs-lag_p-1) 
        S10 = R1 * R0.T / (NumObs-lag_p-1) 
        S11 = R1 * R1.T / (NumObs-lag_p-1) 
        eigValue_lambda, eigvecs = eigh( 
            S10 * S00.I * S01, S11, eigvals_only=False) 
        # 排序特徵向量Eig_vector與特徵值lambda 
        sort_ind = np.argsort(-eigValue_lambda) 
        eigValue_lambda = eigValue_lambda[sort_ind] 
        eigVecs = eigvecs[:, sort_ind] 
        # 將所有eigenvector同除第一行的總和 
        eigValue_lambda = eigValue_lambda.reshape(len(eigValue_lambda), 1) 
        eigVecs_st = eigVecs/np.sum(np.absolute(eigVecs[:, 0][0:2])) 
        # Beta 
        jci_b = eigVecs[:, 0][0:2].reshape(NumDim, 1) 
        jci_beta = eigVecs_st[:, 0][0:2].reshape(NumDim, 1) 
        # Alpha 
        a = np.mat(eigVecs_st[:, 0]) 
        jci_a = S01 * a.T 
        jci_alpha = jci_a/np.sum(np.absolute(jci_a)) 
        # 初始化 c0, d0, c1, d1 
        c0, d0 = 0, 0 
        c1, d1 = np.zeros([NumDim, 1]), np.zeros([NumDim, 1]) 
 
        # 計算 c0, d0, c1, d1，與殘差及VEC項的前置 
        if model_type == 1: 
            W = dY - Ys * jci_beta * jci_alpha.T 
            P = dX.I * W  # [B1,...,Bq] 
            P = P.T 
            cvalue = [12.3329, 4.1475] 
        elif model_type == 2: 
            c0 = eigVecs_st[-1, 0:1] 
            W = dY - (Ys[:, 0:2] * jci_beta + 
                      np.matlib.repmat(c0, NumObs-lag_p-1, 1)) * jci_alpha.T 
            P = dX.I * W  # [B1,...,Bq] 
            P = P.T 
            cvalue = [20.3032, 9.1465] 
        elif model_type == 3: 
            W = dY - Ys * jci_beta * jci_alpha.T

            P = dX.I * W 
            P = P.T 
            c = P[:, -1] 
            c0 = jci_alpha.I * c 
            c1 = c - jci_alpha * c0 
            cvalue = [15.4904, 3.8509] 
        elif model_type == 4: 
            d0 = eigVecs[-1, 0:1] 
            W = dY - (Ys[:, 0:2] * jci_beta + np.arange(1, NumObs-lag_p, 
                                                        1).reshape(NumObs-lag_p-1, 1) * d0) * jci_alpha.T 
            P = dX.I * W 
            P = P.T 
            c = P[:, -1] 
            c0 = jci_alpha.I * c 
            c1 = c - jci_alpha * c0 
            cvalue = [25.8863, 12.5142] 
        elif model_type == 5: 
            W = dY - Ys * jci_beta * jci_alpha.T 
            P = dX.I * W  # [B1,...,Bq] 
            P = P.T 
            c = P[:, -2] 
            c0 = jci_alpha.I * c 
            c1 = c - jci_alpha * c0 
            d = P[:, -1] 
            d0 = jci_alpha.I * d 
            d1 = d - jci_alpha * d0 
            cvalue = [18.3837, 3.8395] 
        # 計算殘差 
        ut = W - dX * P.T 
        Ct_all = jci_a*c0 + c1 + jci_a*d0 + d1 
 
        # 計算VEC項 
        gamma = [] 
        for bi in range(1, lag_p+1): 
            Bq = P[:, (bi-1)*NumDim: bi * NumDim] 
            gamma.append(Bq) 
        temp1 = np.dot(np.dot(jci_b.transpose(), S11[0:2, 0:2]), jci_b) 
        omega_hat = S00[0:2, 0:2] - \
            np.dot(np.dot(jci_a, temp1), jci_a.transpose()) 
        # 把Ct統整在一起 
        Ct = [] 
        Ct.append(c0) 
        Ct.append(d0) 
        Ct.append(c1) 
        Ct.append(d1) 
        self.list_Ct.append(Ct) 
        self.list_jci_alpha.append(jci_alpha) 
        self.list_jci_beta.append(jci_beta) 
        self.list_ut.append(ut) 
        self.list_gamma.append(gamma) 
         
        TraceTest_H = [] 
        TraceTest_T = [] 
        #print("eig_labd", 1-eigValue_lambda) 
        for rn in range(0, NumDim): 
            eig_lambda = np.cumprod(1-eigValue_lambda[rn:NumDim, :]) 
            trace_stat = -2 * np.log(eig_lambda[-1] ** ((NumObs-lag_p-1)/2)) 
            TraceTest_H.append(cvalue[rn] < trace_stat) 
            TraceTest_T.append(trace_stat) 
        return TraceTest_H, ut, 
 
    def Johansen_mean(self, lagp, NumDim=2): 
        # 論文中的closed form mean 
        # lagp指的是VECM的LAG期數 
        B = np.zeros([2, 1]) 
        #print("list of gamma :", self.list_gamma) 
        sumgamma = np.zeros([NumDim, NumDim]) 
        alpha = self.list_jci_alpha[self.optimal_model] 
        beta = self.list_jci_beta[self.optimal_model] 
        mu = self.list_Ct[self.optimal_model] 
        gamma = self.list_gamma[self.optimal_model] 
        # print("alpha :", alpha) 
        # print("beta :", beta) 
        # print("gamma :", gamma) 
        # print("mu : ", mu) 
        B[:, 0] = pd.DataFrame(beta).stack() 
        # 將共整合係數標準化，此為資金權重Capital Weight 
        self.CapitW[:, 0] = B[:, 0] / np.sum(np.absolute(B[:, 0])) 
        for i in range(0, lagp): 
            sumgamma = sumgamma+gamma[i] 
        GAMMA = np.eye(NumDim) - sumgamma 
        # 計算正交化的alpha,beta 
        alpha_orthogonal = alpha.copy() 
        alpha_t = alpha.transpose() 
        alpha_orthogonal[1, 0] = (-(alpha_t[0, 0] * 
                                    alpha_orthogonal[0, 0])) / alpha_t[0, 1] 
        alpha_orthogonal = alpha_orthogonal/sum(abs(alpha_orthogonal)) 
        beta_orthogonal = beta.copy() 
        beta_t = beta.transpose() 
        beta_orthogonal[1, 0] = - \
            ((beta_t[0, 0]*beta_orthogonal[0, 0])) / beta_t[0, 1] 
        beta_orthogonal = beta_orthogonal/sum(abs(beta_orthogonal)) 
 
        # 計算MEAN 
        temp1 = np.linalg.inv( 
            np.dot(np.dot(alpha_orthogonal.transpose(), GAMMA), beta_orthogonal)) 
        C = np.dot(np.dot(beta_orthogonal, temp1), 
                   alpha_orthogonal.transpose()) 
        temp2 = np.linalg.inv(np.dot(alpha.transpose(), alpha)) 
        alpha_hat = np.dot(alpha, temp2) 
        temp3 = np.dot(GAMMA, C) - np.eye(NumDim) 
        C0 = np.mat(mu[0]) 
        C1 = np.mat(mu[2]) 
        D0 = np.mat(mu[1]) 
        D1 = np.mat(mu[3]) 
        C0 = alpha*C0 + C1 + alpha*D0 + D1 
        Ct = alpha*D0 + D1 
        self.Johansen_mu = np.dot( 
            np.dot(alpha_hat.transpose(), temp3), C0)[0, 0] 
        self.Johansen_slope = np.dot( 
            np.dot(alpha_hat.transpose(), temp3), Ct)[0, 0] 
 
    def Johansen_std_correct(self, lag_p, rank=1): 
        # 論文中的closed form std 
        alpha = self.list_jci_alpha[self.optimal_model] 
        beta = self.list_jci_beta[self.optimal_model] 
        ut = self.list_ut[self.optimal_model] 
        mod_gamma = self.list_gamma[self.optimal_model] 
        NumDim = 2 
        if lag_p > 0: 
            # 建立～A 
            tilde_A_11 = alpha 
            tilde_A_21 = np.zeros([NumDim*lag_p, 1]) 
            tilde_A_12 = np.zeros([NumDim, NumDim*lag_p]) 
 
            # 建立～B 
            tilde_B_11 = beta 
            # tilde_A_21與tilde_B_21為相同維度的0矩陣，不重複建立變數 
            tilde_B_3 = np.zeros([NumDim + NumDim*lag_p, NumDim*lag_p]) 
 
            # 用同一個迴圈同時處理～A與～B 
            for qi in range(lag_p): 
                tilde_A_12[0:NumDim, qi*NumDim:(qi+1)*NumDim] = mod_gamma[qi] 
                tilde_B_3[qi*NumDim:NumDim*(2+qi), qi*NumDim:(qi+1) * 
                          NumDim] = np.vstack([np.eye(NumDim), -np.eye(NumDim)]) 
            tilde_A_22 = np.eye(NumDim*lag_p) 
            tilde_A = np.hstack( 
                [np.vstack([tilde_A_11, tilde_A_21]),  np.vstack([tilde_A_12, tilde_A_22])]) 
            tilde_B = np.hstack( 
                [np.vstack([tilde_B_11, tilde_A_21]), tilde_B_3]) 
        elif lag_p == 0: 
            tilde_A = alpha 
            tilde_B = beta 
        tilde_Sigma = np.zeros([NumDim*(lag_p+1), NumDim*(lag_p+1)]) 
        tilde_Sigma[0:NumDim, 0:NumDim] = np.dot( 
            ut.transpose(), ut)/(len(ut)-1) 
        tilde_J = np.zeros([1, 1+NumDim*(lag_p)]) 
        tilde_J[0, 0] = 1 
        if lag_p == 0: 
            temp1 = np.eye(rank)+np.dot(beta.transpose(), alpha) 
            temp2 = np.kron(temp1, temp1) 
            temp3 = np.linalg.inv(np.eye(rank)-temp2) 
            omega = np.dot(ut.transpose(), ut)/(len(ut)-1) 
            temp4 = np.dot(np.dot(beta.transpose(), omega), beta) 
            var = np.dot(temp3, temp4) 
        else: 
            temp1 = np.eye(NumDim*(lag_p+1)-1) + np.dot(tilde_B.transpose(), tilde_A) 
            temp2 = np.kron(temp1, temp1) 
            k = (NumDim*(lag_p+1)-1)*(NumDim*(lag_p+1)-1) 
            temp3 = np.linalg.inv(np.eye(k)-temp2) 
            temp4 = np.dot(np.dot(tilde_B.transpose(), tilde_Sigma), tilde_B) 
            temp4 = temp4.flatten('F') 
            temp5 = np.dot(temp3, temp4) 
            sigma_telta_beta = np.zeros( 
                [NumDim*(lag_p+1)-1, NumDim*(lag_p+1)-1]) 
            for i in range(NumDim*(lag_p+1)-1): 
                for j in range(NumDim*(lag_p+1)-1): 
                    sigma_telta_beta[i][j] = temp5[0, i+j*(NumDim*(lag_p+1)-1)] 
            var = np.dot(np.dot(tilde_J, sigma_telta_beta), 
                         tilde_J.transpose()) 
        self.Johansen_stdev = np.sqrt(var)[0, 0] 
 
    def execute(self): 
        try :
            if not self._whitenoise(self._data): 
                return 0, 0, 0, 0, 0
            else: 
                #print('successsssssssssssssssssssssssssssss')
                self._JCI_AutoSelection(self._data, self.p-1) 
                self.Johansen_mean(self.p-1) 
                self.Johansen_std_correct(self.p-1) 
                return [self.Johansen_mu, self.Johansen_stdev, self.optimal_model+1, self.CapitW[0, 0], self.CapitW[1, 0]]
        except : 
            #print('exception')
            return 0, 0, 0, 0, 0