import networkx, scipy, copy, math, multiprocessing
import numpy as np
from time import time
from networkx.algorithms.components.connected import connected_components
import statsmodels.api as sm
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.cluster import KMeans
import wingman as wm
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import random
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def quasi_oracle(oriZ, oriX, oriY, real_labels, est_labels):

    N, qq = oriZ.shape
    quasi_labels = np.zeros(N)
    real_knum = len(np.unique(real_labels))
    est_knum = len(np.unique(est_labels))

    for k in range(est_knum):
        location_k = np.where(est_labels == k)[0]
        the_k = np.bincount((real_labels[location_k]).astype(int)).argmax()
        quasi_labels[location_k] = the_k

    return wm.oracle(oriZ, oriX, oriY, quasi_labels)


def createATA(oriZ):
    N, qq = oriZ.shape
    a = np.kron(np.ones(N), np.identity(qq)).T
    return N * np.identity(N * qq) - np.dot(a, a.T)


def block_diag_matrix(oriZ): 
    input_array = oriZ

    num_rows, num_cols = input_array.shape

    block_diag_rows = num_rows
    block_diag_cols = num_rows * num_cols
    block_diag_matrix = np.zeros((block_diag_rows, block_diag_cols))

    for i in range(num_rows):
        block_diag_matrix[i, i * num_cols:(i + 1) * num_cols] = input_array[i]

    return block_diag_matrix


def least_squares_fit_grouping_group_mode(oriZ, oriX, oriY):
    N = len(oriZ)
    oriY = oriY.reshape((N, 1))
    if oriX is not None:
        data = np.hstack((oriZ, oriX, oriY))
    else:
        data = np.hstack((oriZ, oriY))
    X = data[:, :-1]  # 倒数第二列之前的列为自变量
    y = data[:, -1]  # 倒数第二列为因变量
    model = sm.OLS(y, X).fit()
    coef = model.params
    # 将回归系数存储到数组中
    if oriX is not None:
        return coef[:oriZ.shape[1]], coef[oriZ.shape[1]:-1]
    else:
        return coef[:oriZ.shape[1]], np.zeros(oriX.shape[1])


def least_squares_fit(oriZ, oriX, oriY):
    N = len(oriZ)
    oriY = oriY.reshape((N, 1))
    if oriX is not None:
        data = np.hstack((oriZ, oriX, oriY))
    else:
        data = np.hstack((oriZ, oriY))
    X = data[:, :-1]  # 倒数第二列之前的列为自变量
    y = data[:, -1]  # 倒数第二列为因变量
    model = sm.OLS(y, X).fit()
    coef = model.params
    # 将回归系数存储到数组中

    return coef[:oriZ.shape[1]]


def least_squares_fit_group(oriZ, oriX, oriY, estlabel):
    N = len(oriZ)
    oriY = oriY.reshape((N, 1))

    if oriX is not None:
        data = np.hstack((oriZ, oriX, oriY))
    else:
        data = np.hstack((oriZ, oriY))
    groups = np.unique(estlabel)
    coefficients = np.empty((len(groups), data.shape[1] - 1))
    for i, group in enumerate(groups):
        group_data = data[estlabel == group]
        X = group_data[:, :-1]
        y = group_data[:, -1]  
        model = sm.OLS(y, X).fit()
        coef = model.params
        coefficients[i] = coef
    return coefficients



def local_regression(data, KNN_num):
    results = []
    dist_matrix = squareform(pdist(data))

    for i in range(len(data)):
        distances = dist_matrix[i]
        nearest_indices = np.argsort(distances)[1:KNN_num + 1]

        nearest_data = [data[j] for j in nearest_indices]

        params = perform_local_regression(nearest_data)

        results.append(params)

    return np.array(results)


def perform_local_regression(nearest_data):
    X = np.array([data[:-1] for data in nearest_data])
    y = np.array([data[-1] for data in nearest_data])

    model = sm.OLS(y, X).fit()

    return model.params


def knn_regression(oriZ, oriX, oriY, KNN_num):
    N, qq = oriZ.shape
    if oriX is not None:
        oriY = oriY.reshape((N, 1))
        data = np.hstack((oriZ, oriX, oriY))
    else:
        oriY = oriY.reshape((N, 1))
        data = np.hstack((oriZ, oriY))
    if KNN_num is None:
        KNN_num = 10
    results = local_regression(data, KNN_num)

    ini_est_theta = results[:, :qq].ravel()
    ini_est_beta = np.mean(results[:, qq:], axis=0)
    inik = None
    Coefficients = None
    estlabel = None

    return ini_est_theta, ini_est_beta, inik, Coefficients, estlabel


def label_to_label(labels):
    the_real = np.zeros(len(labels))
    the_real_unique = np.sort(np.unique(labels))
    for i in range(len(labels)):
        the_real[i] = np.where(the_real_unique == labels[i])[0][0]
    return the_real


def kmeans_and_least_squares_fit(Cluster_objects, oriZ, oriX, oriY, Inik=None):
    N, qq = oriZ.shape
    if oriX is not None:
        pp = oriX.shape[1]
    else:
        pp = 0
    if Inik is None:

        inik = math.floor(np.sqrt(N) / 3)
        if inik <= 6:
            inik = 6
        if inik >= 10:
            inik = math.floor(np.sqrt(N))

    else:
        inik = Inik
    print('inik is:', inik)
    kmeans = KMeans(n_clusters=inik,
                    random_state=0,
                    n_init="auto",
                    init='k-means++').fit(Cluster_objects)
    # kmeans =SpectralClustering(n_clusters=inik,random_state=0,assign_labels='cluster_qr').fit(Cluster_objects)

    estlabel = kmeans.labels_
    ini_kforeachgroup = np.zeros(inik)
    for i in range(inik):
        ini_kforeachgroup[i] = np.sum(estlabel == i)
    print('ini_kforeachgroup:', ini_kforeachgroup)
    Coefficients = least_squares_fit_group(oriZ, oriX, oriY, estlabel)

    coefficients = np.zeros((N, pp + qq))
    for i in range(N):
        coefficients[i] = Coefficients[estlabel[i]]

    return coefficients[:, :qq], np.mean(coefficients[:, qq:],
                                         axis=0), inik, Coefficients, estlabel


def split_arrays_randomly(arrays, H):
    assert all(len(arr) == len(arrays[0])
               for arr in arrays), "All arrays must have the same length"

    length = len(arrays[0])

    indices = np.arange(length)
    new_indices = np.random.shuffle(indices)

    chunk_size = length // H
    remainder = length % H

    chunks = [[] for _ in range(H)]
    start = 0

    for i in range(H):
        end = start + chunk_size + (1 if i < remainder else 0)
        for arr in arrays:
            chunks[i].append(arr[new_indices[start:end]])
        start = end

    return chunks, new_indices


def ridge(oriZ, oriX, oriY, lamstar=0.001):

    Z = block_diag_matrix(oriZ)
    N, qq = oriZ.shape
    if oriX is not None:
        pp = oriX.shape[1]
    else:
        pp = 0
    ATA = createATA(oriZ)

    if oriX is not None:
        XTX_1 = np.linalg.inv(np.dot(oriX.T, oriX))
        XTX_1XT = np.dot(XTX_1, oriX.T)
        Q_X = np.identity(N) - np.dot(oriX, XTX_1XT)
    else:
        Q_X = np.identity(N)
    ZT = Z.transpose()
    ZTQ_XZ = (ZT.dot(Q_X)).dot(Z)
    M1 = np.linalg.inv(ZTQ_XZ + lamstar * ATA)
    M2 = (ZT.dot(Q_X)).dot(oriY)

    esttheta = np.dot(M1, M2)
    if oriX is not None:
        estbeta = np.dot(XTX_1XT, oriY - Z.dot(esttheta))
    else:
        estbeta = np.zeros(pp)
    return esttheta, estbeta


def least_squares_fit_group_return(oriZ, oriX, oriY, estlabel):
    qq = oriZ.shape[1]
    estlabel = estlabel.astype(int)
    coefficients = least_squares_fit_group(oriZ, oriX, oriY, estlabel)
    Esttheta = np.zeros((len(estlabel), oriZ.shape[1]))
    for i in range(len(estlabel)):
        Esttheta[i] = coefficients[estlabel[i]][:qq]
    return Esttheta, coefficients[:, :qq], np.mean(coefficients[:, qq:],
                                                   axis=0)


class subgroups_analysis():

    def __init__(self,
                 oriZ,
                 oriY,
                 oriX=None,
                 Feture_name_Z=None,
                 AN=None,
                 ANT=None,
                 ATA=None,
                 ini_esttheta=None,
                 lam_list=[0.1, 0.5, 1, 5],
                 real_label=None,
                 real_beta=None,
                 real_theta=None,
                 vertheta=3,
                 gamma=1,
                 doorsill=0.01,
                 tolerance=0.001,
                 T0=50,
                 T1=100,
                 c1=1,
                 get_res=1,
                 iniway='kmeans',
                 inik=None,
                 hot_start=False,
                 incidence_matrix=None,
                 print_self=False,
                 ss_analysis=False,
                 intercept_hete=False,
                 real_pp=None,
                 real_qq=None):

        self.intercept_hete = intercept_hete

        self.print_self = print_self

        self.oriZ = oriZ
        self.oriX = oriX
        N = oriZ.shape[0]
        self.oriY = oriY.reshape(N)

        self.all_train_switch = False
        self.best_lambda = None
        self.Feture_name_Z = Feture_name_Z

        self.lamstar = 0.0000001
        self.iniway = iniway
        self.inik = inik

        self.N, self.qq = self.oriZ.shape
        if oriX is not None:
            self.pp = oriX.shape[1]
        else:
            self.pp = 0

        if ini_esttheta is None and ss_analysis is False:
            ini_Esttheta, ini_est_beta, inik, KmCoe, KmLabel = self._iniest_single(
            )
            self.ini_esttheta = ini_Esttheta.flatten()

            self.ini_est_beta = ini_est_beta
            self.inik = inik
            self.KmCoe = KmCoe
            self.KmLabel = KmLabel
        else:
            self.ini_esttheta = ini_esttheta

        if AN is None and ss_analysis is False:
            self.incidence_matrix, self.AN = wm.createA(self.oriZ)
            self.ANT = self.AN.transpose()
            self.ATA = wm.createATA(self.oriZ)
        else:
            self.incidence_matrix = incidence_matrix
            self.AN = AN
            self.ANT = ANT
            self.ATA = ATA

        self.lam_list = lam_list

        self.real_label = real_label
        self.real_beta = real_beta
        if self.oriX is None:
            self.real_beta = None
        self.real_theta = real_theta

        self.vertheta = vertheta
        self.gamma = gamma
        self.doorsill = doorsill
        self.tolerance = tolerance
        self.T0 = T0
        self.T1 = T1
        self.c1 = c1
        self.get_res = get_res
        self.hot_start = hot_start

    def __str__(self):
        """
        Description of the model.
        :return: String representation of the model
        """

        self.name = f"SA(lam={self.best_lambda}, iniway={self.iniway,self.inik})"

        if self.real_label is not None:
            self.real_K = np.unique(self.real_label).shape[0]

        description = "Model:        %s\t\n" % self.name
        if self.all_train_switch:

            description += "Size:        %s\n" % f"(N,q,p)=({self.N,self.qq,self.pp})"
            # description += "Variance:  %s\n" % f"(piy,piy,(piy)/(pix+piy))=({self.piy,self.pix,self.piy/(self.piy+self.pix)})"
            if self.real_label is not None:
                description += "Real.K&Est.K:  %s\n" % f"({self.real_K},{self.knum})"
            else:
                description += "Est.K:     %s\n" % f"{self.knum}"
            description += "Est.EachNum:  %s\n" % f"({self.kforeachgroup_stage1})"
            description += "RI:            %s\n" % f"{self.RI}"
            description += "FDR:            %s\n" % f"{self.FDR}"
            description += "TPR:            %s\n" % f"{self.TPR}"
            if self.real_beta is not None:
                description += "MSE_beta:    %s\n" % self.mse_beta
                description += "Oracle_MSE_beta: %s\n" % self.oracle_mse_beta
            if self.real_theta is not None:
                description += "MSE_theta:   %s\n" % self.mse_theta
                description += "Oracle_MSE_theta: %s\n" % self.oracle_mse_theta
            description += "R_squared:   %s\n" % self.r_squared_manual
            description += "se:   %s\n" % self.se_rounded
            description += "pvalue:   %s\n" % self.pvalue_rounded
            description += "BIC:             %s\n" % self.bic
            description += "Run_time:             %s\n" % self.run_time
        else:
            description += "Not trained yet"
        return description



    def _iniest_single_ridge(self):
        ini_est_theta, ini_est_beta = ridge(self.oriZ, self.oriX, self.oriY,
                                            self.lamstar)
        return ini_est_theta

    def _iniest_single(self):


        if self.iniway == 'ridge':
            print('iniway is ridge')
            ini_est_theta, ini_est_beta = ridge(self.oriZ, self.oriX,
                                                self.oriY, self.lamstar)
            inik, KmCoe, KmLabel = None, None, None

        elif self.iniway == 'kmeans':
            print('iniway is kmeans')
            ini_est_theta = self._iniest_single_ridge()

            Cluster_objects = ini_est_theta.reshape(self.oriZ.shape)

            ini_est_theta, ini_est_beta, inik, KmCoe, KmLabel = kmeans_and_least_squares_fit(
                Cluster_objects, self.oriZ, self.oriX, self.oriY, self.inik)
        elif self.iniway == 'knn':
            print('iniway is knn')
            ini_est_theta, ini_est_beta, inik, KmCoe, KmLabel = knn_regression(
                self.oriZ, self.oriX, self.oriY, self.inik)
        else:
            ini_est_theta, ini_est_beta = ridge(self.oriZ, self.oriX,
                                                self.oriY, self.lamstar)
            inik, KmCoe, KmLabel = None, None, None

        return ini_est_theta, ini_est_beta, inik, KmCoe, KmLabel

    def _analysis(self, Lam0=0.1, once=False, doorsill=None):
        if doorsill is None:
            doorsill = self.doorsill

        starttime = time()

        Z = wm.block_diag_matrix(self.oriZ)
        # print('Z:',Z.shape)

        # 初值
        if self.oriX is not None:
            XTX_1 = np.linalg.inv(np.dot(self.oriX.T, self.oriX))

            XTX_1XT = np.dot(XTX_1, self.oriX.T)
            Q_X = np.identity(self.N) - np.dot(self.oriX, XTX_1XT)
        else:
            Q_X = np.identity(self.N)
        ZT = Z.transpose()
        ZTQ_XZ = (ZT.dot(Q_X)) * (Z)
        ZTQY = ZT.dot(Q_X).dot(self.oriY)
        # print((ZTQ_XZ+self.vertheta*self.ATA).shape)
        M1 = np.linalg.inv(ZTQ_XZ + self.vertheta * self.ATA)

        vdelta = self.AN.dot(self.ini_esttheta)
        vnu = np.zeros(vdelta.shape)

        counter = 0
        gap = 0

        while counter <= self.T1:
            esttheta = np.array(
                np.dot(
                    M1,
                    ZTQY.reshape(-1) +
                    self.vertheta * self.ANT.dot(vdelta -
                                                 (1 / self.vertheta) * vnu)))
            if esttheta.shape[0] == 1:
                esttheta = esttheta[0]

            if self.oriX is not None:
                # print(Z.shape,esttheta.shape)
                estbeta = np.dot(XTX_1XT, self.oriY - Z.dot(esttheta))
            else:
                estbeta = np.zeros(self.pp)

            va = self.AN.dot(esttheta)

            vxi = va + (1 / self.vertheta) * vnu
            xii = vxi.reshape((int(len(vxi) / self.qq), self.qq))
            delta = wm.MCP(xii, self.gamma, Lam0, self.vertheta)
            vdelta = delta.flatten()

            vnu = vnu + self.vertheta * (va - vdelta)
            counter = counter + 1
            gap = np.linalg.norm(va - vdelta, ord=2)
            if counter % 1000 == 0:
                pass
            if counter > self.T0 and gap < self.tolerance:
                break

        endtime = time()
        run_time = endtime - starttime

        self.oriEsttheta = esttheta.reshape(self.oriZ.shape)
        dist_matrix = squareform(pdist(self.oriEsttheta))
        lower_indices = np.tril_indices(dist_matrix.shape[0], -1)
        dist_matrix[lower_indices] = 100
        fimatrix = dist_matrix
        Dh = wm.toZ(fimatrix, self.N, doorsill)

        grouph = wm.Group(fimatrix, doorsill)
        kforeachgroup_stage1 = np.zeros(len(grouph))
        grouh = []
        for i in range(len(grouph)):
            grouh.append(np.array(list(grouph[i])).astype(int))
            kforeachgroup_stage1[i] = len(grouph[i])

        samplelabel = np.zeros(self.N)
        for i in range(len(grouph)):
            samplelabel[grouh[i]] = i

        knum = np.unique(samplelabel).shape[0]

        if self.get_res == 0:
            fiesttheta = esttheta
            fiestbeta = estbeta
            Esttheta = fiesttheta.reshape(self.oriZ.shape)

            invD = np.linalg.inv(np.dot(Dh.T, Dh))
            invDDT = np.dot(invD, Dh.T)
            Alpha = np.dot(invDDT, Esttheta)
            Esttheta = np.dot(Dh, Alpha)

        elif self.get_res == 1:
            Esttheta, Alpha, fiestbeta = least_squares_fit_group_return(
                self.oriZ, self.oriX, self.oriY, samplelabel)

        elif self.get_res == 2:
            if knum > 1:
                fiestbeta, Alpha, Esttheta = wm.oracle(self.oriZ, self.oriX,
                                                       self.oriY, samplelabel)
            else:
                fiestbeta = estbeta
                Esttheta = fiesttheta.reshape(self.oriZ.shape)
                Alpha = np.mean(Esttheta, axis=0)

        if self.real_label is not None:
            RI = rand_score(self.real_label, samplelabel)
            ARI = adjusted_rand_score(self.real_label, samplelabel)
            Cm1 = pair_confusion_matrix(self.real_label, samplelabel)
            np.seterr(divide='ignore', invalid='ignore')
            FDR = Cm1[0, 1] / (Cm1[0, 1] + Cm1[1, 1]
                               )  #False Discovery Rate 不该在一组的被分到了一组
            TPR = Cm1[1, 1] / (Cm1[1, 1] + Cm1[1, 0])  #True Positive Rate
        else:
            RI = None
            ARI = None
            TPR = None
            Cm1 = None
            FDR = None
            pipeline_cm = None

        resx = np.ones(self.N)
        for i in range(self.N):
            resx[i] = np.dot(self.oriZ[i], Esttheta[i])
        y_preds = np.zeros(self.N)
        if self.oriX is None:
            y_preds = resx
            Residual = self.oriY - y_preds
        else:
            y_preds = resx + np.dot(self.oriX, fiestbeta)
            Residual = self.oriY - y_preds

        bic = (np.log(np.dot(Residual.T, Residual) / self.N) + self.c1 *
               (np.log(self.N * self.qq + self.pp)) *
               (np.log(self.N) / self.N) * (knum * self.qq + self.pp))

        self.train_switch = True

        if once is False:
            result = [
                RI, FDR, Cm1, TPR, Alpha, fiestbeta, Esttheta, knum,
                samplelabel, Dh, grouph, kforeachgroup_stage1, run_time,
                y_preds, Residual, bic, Lam0
            ]
            return result

        else:
            self.all_train_switch = True

            self.RI, self.FDR, self.Cm1, self.TPR, self.Alpha, self.fiestbeta, self.Esttheta, self.knum, self.samplelabel, self.Dh, self.grouph, self.kforeachgroup_stage1, self.run_time, self.y_preds, self.Residual, self.bic = RI, FDR, Cm1, TPR, Alpha, fiestbeta, Esttheta, knum, samplelabel, Dh, grouph, kforeachgroup_stage1, run_time, Residual, bic

            if self.real_beta is not None:
                self.mse_beta = np.mean((self.real_beta - self.ini_estbeta)**2)
            if self.real_theta is not None:
                self.mse_theta = np.mean(
                    (self.Esttheta.flatten() - self.real_theta.flatten())**2)

    def _calculate_summary_statiscs(self):
        fiKnum = self.knum
        residuals = self.oriY - self.y_preds
        rmse = np.sqrt(mean_squared_error(self.oriY, self.y_preds))
        mae = mean_absolute_error(self.oriY, self.y_preds)
        r_squared_manual = r2_score(self.oriY, self.y_preds)
        X = self.oriX
        diagZ = wm.block_diag_matrix(self.oriZ)
        degree_of_freedom_Z = self.N - self.qq * self.fiKnum - self.pp
        sigma_hat_squared_Z = (residuals @ residuals) / degree_of_freedom_Z

        indicator_matrix = np.kron(
            np.eye(fiKnum)[np.array(self.samplelabel).astype(int)],
            np.eye(self.qq))
        Z = diagZ.dot(indicator_matrix)
        if self.oriX is not None:
            XTX_inv = np.linalg.inv(X.T.dot(X))
            X_XTX_inv_XT = X.dot(XTX_inv).dot(X.T)
            I_X_XTX_inv_XT = np.eye(self.N) - X_XTX_inv_XT
        else:
            I_X_XTX_inv_XT = np.eye(self.N)

        ZT_I_X_XTX_inv_XT_Z = Z.T.dot(I_X_XTX_inv_XT).dot(Z)
        ZT_I_X_XTX_inv_XT_Z_inv = np.linalg.inv(ZT_I_X_XTX_inv_XT_Z)
        standard_errors_Z = np.sqrt(
            np.diag(sigma_hat_squared_Z * ZT_I_X_XTX_inv_XT_Z_inv)).reshape(
                self.Alpha.shape)

        t_value_Z = self.Alpha / standard_errors_Z

        p_value_Z = 2 * (1 -
                         stats.t.cdf(np.abs(t_value_Z), degree_of_freedom_Z))

        if self.oriX is not None:
            degree_of_freedom_X = self.N - self.qq * self.fiKnum - self.pp
            sigma_hat_squared_X = (residuals @ residuals) / degree_of_freedom_X

            ZTZ_inv = np.linalg.inv(Z.T.dot(Z))
            Z_ZTZ_inv_ZT = Z.dot(ZTZ_inv).dot(Z.T)
            I_Z_ZTZ_inv_ZT = np.eye(len(Z)) - Z_ZTZ_inv_ZT
            XT_I_Z_ZTZ_inv_ZT_X = X.T.dot(I_Z_ZTZ_inv_ZT).dot(X)
            XT_I_Z_ZTZ_inv_ZT_X_inv = np.linalg.inv(XT_I_Z_ZTZ_inv_ZT_X)

            standard_errors_X = np.sqrt(
                np.diag(sigma_hat_squared_X * XT_I_Z_ZTZ_inv_ZT_X_inv))

            t_value_X = self.fiestbeta / standard_errors_X

            p_value_X = 2 * (
                1 - stats.t.cdf(np.abs(t_value_X), degree_of_freedom_X))
        else:
            standard_errors_X = None
            t_value_X = None
            p_value_X = None

        return r_squared_manual, standard_errors_Z, t_value_Z, p_value_Z, standard_errors_X, t_value_X, p_value_X


    def _grouping(self, doorsill=None, refresh=False):
        if doorsill is None:
            doorsill = self.doorsill
        dist_matrix = squareform(pdist(self.oriEsttheta))
        lower_indices = np.tril_indices(dist_matrix.shape[0], -1)
        # 将下三角部分的值设置为 100
        dist_matrix[lower_indices] = 100
        fimatrix = dist_matrix
        # fimatrix=wm.VtoM(Delta)
        Dh = wm.toZ(fimatrix, self.N, doorsill)

        grouph = wm.Group(fimatrix, doorsill)
        kforeachgroup_stage1 = np.zeros(len(grouph))
        grouh = []  #将set元素变为array元素
        for i in range(len(grouph)):
            grouh.append(np.array(list(grouph[i])).astype(int))
            kforeachgroup_stage1[i] = len(grouph[i])

        samplelabel = np.zeros(self.N)  #给对应样本打上label
        for i in range(len(grouph)):
            samplelabel[grouh[i]] = i

        knum = np.unique(samplelabel).shape[0]


    def _collect_result(self, index, result, results, order):
        results[index] = result
        order.append((index, result[-1]))
        RI, FDR, Cm1, TPR, Alpha, fiestbeta, Esttheta, knum, samplelabel, Dh, grouph, kforeachgroup_stage1, run_time, y_preds, Residual, bic, Lam0 = result
        print('Lam:', Lam0, 'bic:', bic, 'Knum:', knum, 'K_for_eachg:',
              kforeachgroup_stage1, 'run_time:', run_time)

    def _sort_results(self, results, order):
        Lambda = [result[-1] for result in results]
        order = np.argsort(Lambda)
        sorted_results = [results[i] for i in order]
        return sorted_results

    def _single_sa(self,
                   lam_list=None,
                   parallel=False,
                   doorsill=None,
                   asyn=True):
        if doorsill is None:
            doorsill = self.doorsill
        if lam_list is None:
            lam_list = self.lam_list

        # _bic_selection

        lam_count = len(lam_list)
        BIC_list = np.zeros(lam_count)
        Knum_list = [0] * lam_count
        RI_list = [0] * lam_count

        results = [0] * lam_count
        if len(lam_list) == 1:
            parallel = False

        if parallel is True:

            max_workers = round(multiprocessing.cpu_count() * 0.8)
            if asyn is False:
                with multiprocessing.Pool(processes=max_workers) as pool:
                    results = pool.map(self._analysis, self.lam_list)

            else:
                results = [None] * len(lam_list)
                order = []  

                with multiprocessing.Pool(processes=max_workers) as pool:
                    for index, lam in enumerate(lam_list):
                        pool.apply_async(
                            self._analysis,
                            args=(lam, ),
                            callback=lambda result, index=index: self.
                            _collect_result(index, result, results, order)
                        )

                    pool.close()
                    pool.join()

            sorted_results = self._sort_results(results, order)

            est_theta_list = []

            for i in range(lam_count):
                RI, FDR, Cm1, TPR, Alpha, fiestbeta, Esttheta, knum, samplelabel, Dh, grouph, kforeachgroup_stage1, run_time, y_preds, Residual, bic, Lam0 = sorted_results[
                    i]
                Knum_list[i] = knum
                BIC_list[i] = bic
                RI_list[i] = RI
                est_theta_list.append(Esttheta)
        else:
            results = [None] * len(lam_list)
            est_theta_list = []
            for i in range(lam_count):
                Lam0 = lam_list[i]
                results[i] = self._analysis(Lam0)

                RI, FDR, Cm1, TPR, Alpha, fiestbeta, Esttheta, knum, samplelabel, Dh, grouph, kforeachgroup_stage1, run_time, y_preds, Residual, bic, Lam0 = results[
                    i]

                Knum_list[i] = knum
                BIC_list[i] = bic
                RI_list[i] = RI
                est_theta_list.append(Esttheta)
                if self.hot_start is True:
                    self.ini_esttheta = Esttheta.flatten()
        if self.oriZ.shape[1] == 1:
            self.est_theta_list = np.array(est_theta_list).reshape(
                (lam_count, self.N))
        else:
            self.est_theta_list = np.array(est_theta_list).reshape(
                (lam_count, self.N, self.qq))


        best_index = np.argmin(BIC_list)

        self.best_lambda = lam_list[best_index]
        self.best_result = results[best_index]

        try:
            self.RI, self.FDR, self.Cm1, self.TPR, self.Alpha, self.fiestbeta, self.Esttheta, self.knum, self.samplelabel, self.Dh, self.grouph, self.kforeachgroup_stage1, self.run_time, self.y_preds, self.Residual, self.bic = self.best_result
        except:
            self.RI, self.FDR, self.Cm1, self.TPR, self.Alpha, self.fiestbeta, self.Esttheta, self.knum, self.samplelabel, self.Dh, self.grouph, self.kforeachgroup_stage1, self.run_time, self.y_preds, self.Residual, self.bic, _ = self.best_result

        self.all_train_switch = True
        self.bic_matrix = np.vstack((lam_list, Knum_list, BIC_list)).T

        if self.real_label is not None:
            self.oracle_beta, self.oracle_Zeta, self.oracle_Theta = wm.oracle(
                self.oriZ, self.oriX, self.oriY, self.real_label)
        if self.real_beta is not None:
            self.mse_beta = np.mean((self.real_beta - self.fiestbeta)**2)
            self.oracle_mse_beta = np.mean(
                (self.real_beta - self.oracle_beta)**2)
        if self.real_theta is not None:
            self.mse_theta = np.mean(
                (self.Esttheta.flatten() - self.real_theta.flatten())**2)
            self.oracle_mse_theta = np.mean(
                (self.oracle_Theta.flatten() - self.real_theta.flatten())**2)

        try:
            self.r_squared_manual, self.se_rounded, self.t_value, self.pvalue_rounded, self.se_intercept_rounded, self.t_intercept_value, self.pvalue_intercept_rounded = self._calculate_summary_statiscs(
            )
        except Exception as e:
            print(f"Error calculating summary statistics: {e}")
            self.r_squared_manual = None
            self.se_rounded = None
            self.pvalue_rounded = None
            self.se_intercept_rounded = None
            self.pvalue_intercept_rounded = None
            self.t_value = None
            self.t_intercept_value = None

        if self.print_self is True:
            print('sa', self)

        # 删除self.An
        if hasattr(self, 'AN'):
            del self.AN
        if hasattr(self, 'ANT'):
            del self.ANT
        if hasattr(self, 'ATA'):
            del self.ATA
        if hasattr(self, 'incidence_matrix'):
            del self.incidence_matrix


class Subgroups_analysis(subgroups_analysis):

    def __init__(self,
                 oriZ,
                 oriY,
                 oriX=None,
                 Feture_name_Z=None,
                 AN=None,
                 ANT=None,
                 ATA=None,
                 ini_esttheta=None,
                 lam_list=[0.1, 0.5, 1, 5],
                 real_label=None,
                 real_beta=None,
                 real_theta=None,
                 vertheta=3,
                 gamma=1,
                 doorsill=0.1,
                 tolerance=0.001,
                 T0=50,
                 T1=100,
                 c1=0.5,
                 get_res=1,
                 iniway='kmeans',
                 inik=None,
                 hot_start=False,
                 location=None,
                 redu_way=None,
                 KNN_num=5,
                 intercept_hete=True,
                 seed=None,
                 print_self=False):
        self.real_X = oriX
        if oriX is not None:
            self.real_pp = oriX.shape[1]
        else:
            self.real_pp = 0
        self.real_Z = oriZ
        self.real_qq = oriZ.shape[1]

        if oriX is None and oriZ is None:
            raise ValueError('oriX and oriZ cannot be None at the same time')
        if intercept_hete is True and oriZ is not None:
            oriZ = np.hstack((np.ones((oriZ.shape[0], 1)), oriZ))
        elif intercept_hete is True and oriZ is None:
            oriZ = np.ones((oriY.shape[0], 1))

        if intercept_hete is False and oriX is not None:
            oriX = np.hstack((np.ones((oriX.shape[0], 1)), oriX))
        elif intercept_hete is False and oriX is None:
            oriX = np.ones((oriZ.shape[0], 1))

        if redu_way == 'full':
            redu_way = None

        # if redu_way is not None and location is not None:
        if redu_way is not None:
            incidence_matrix, AN = CreateA(oriZ,
                                           way=redu_way,
                                           ini_Esttheta_or_location=location,
                                           KNN_num=KNN_num,
                                           seed=seed)
            ANT = AN.transpose()
            ATA = wm.dir_createATA(AN)
        else:
            incidence_matrix = None
            AN = None
            ANT = None

        super().__init__(oriZ,
                         oriY,
                         oriX,
                         Feture_name_Z,
                         AN,
                         ANT,
                         ATA,
                         ini_esttheta,
                         lam_list,
                         real_label,
                         real_beta,
                         real_theta,
                         vertheta,
                         gamma,
                         doorsill,
                         tolerance,
                         T0,
                         T1,
                         c1,
                         get_res,
                         iniway,
                         inik=inik,
                         hot_start=hot_start,
                         incidence_matrix=incidence_matrix,
                         print_self=print_self,
                         intercept_hete=intercept_hete,
                         real_pp=self.real_pp,
                         real_qq=self.real_pp)

        self.inik = inik
        self.intercept_hete = intercept_hete
        self.location = location


def mst_incidence_matrix(points, qq):
    """
    计算给定点集的最小生成树，并返回转置后的关联矩阵。
    
    参数:
    points (numpy array): 大小为 (N, p) 的点坐标矩阵，N是点的个数，p是点的维度。
    
    返回:
    incidence_matrix (numpy array): 大小为 (N-1, N) 的关联矩阵。
    """
    N = points.shape[0]

    # Step 1: 计算距离矩阵
    dist_matrix = distance_matrix(points, points)

    # Step 2: 计算最小生成树
    mst = minimum_spanning_tree(dist_matrix).toarray()

    # Step 3: 生成关联矩阵（转置形式）
    edges = np.argwhere(mst > 0)  # 找到最小生成树的边 (i, j)
    incidence_matrix = np.zeros((N - 1, N))  # (N-1) × N 的关联矩阵

    for idx, (i, j) in enumerate(edges):
        incidence_matrix[idx, i] = -1
        incidence_matrix[idx, j] = 1

    A = scipy.sparse.kron(incidence_matrix, scipy.sparse.eye(*qq))

    return incidence_matrix, A


def get_mst_adjacency_matrix(values, adj_matrix, qq):
    n = len(values)

    # 构建加权图
    weighted_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] != 0:  # 判断是否有边
                weight = abs(values[i] - values[j])
                weighted_edges.append((i, j, weight))

    # 构建图并使用 Kruskal 算法计算最小生成树
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)

    # 使用 Kruskal 获取最小生成树
    mst = nx.minimum_spanning_tree(G, algorithm="kruskal")

    # 获取最小生成树的边
    mst_edges = list(mst.edges())

    # 构造关联矩阵 (N-1) x N
    num_edges = len(mst_edges)
    incidence_matrix = np.zeros((num_edges, n))

    for idx, (u, v) in enumerate(mst_edges):
        incidence_matrix[idx, u] = 1
        incidence_matrix[idx, v] = -1

    A = scipy.sparse.kron(incidence_matrix, scipy.sparse.eye(*qq))

    return incidence_matrix, A


def epsilon_graph_incidence_matrix(points, epsilon, qq):

    N = points.shape[0]
    dist_matrix = distance_matrix(points, points)

    # 找到所有符合 epsilon 条件的边
    edges = np.argwhere((dist_matrix < epsilon) & (dist_matrix > 0))  # 排除自身距离

    # 构造关联矩阵
    incidence_matrix = np.zeros((len(edges) // 2, N))
    for idx, (i, j) in enumerate(edges[::2]):  # 避免重复边
        incidence_matrix[idx, i] = -1
        incidence_matrix[idx, j] = 1

    A = scipy.sparse.kron(incidence_matrix, scipy.sparse.eye(*qq))

    return incidence_matrix, A


def knn_graph_incidence_matrix(points, k, qq):

    N = points.shape[0]
    dist_matrix = distance_matrix(points, points)

    knn_indices = np.argsort(dist_matrix, axis=1)[:, 1:k + 1]  # 排除自身

    edges = []
    for i in range(N):
        for j in knn_indices[i]:
            if i < j:  
                edges.append((i, j))


    incidence_matrix = np.zeros((len(edges), N))
    for idx, (i, j) in enumerate(edges):
        incidence_matrix[idx, i] = -1
        incidence_matrix[idx, j] = 1

    A = scipy.sparse.kron(incidence_matrix, scipy.sparse.eye(*qq))

    return incidence_matrix, A


def random_spanning_tree_incidence_matrix(graph, qq):
    N = graph.shape[0]
    visited = set()
    tree_edges = []

    start_node = random.choice(range(N))
    visited.add(start_node)

    while len(visited) < N:
        edges = [(i, j) for i in visited for j in range(N)
                 if j not in visited and graph[i, j] > 0]
        if not edges:
            break  
        edge = random.choice(edges)
        tree_edges.append(edge)
        visited.add(edge[1])

    num_edges = len(tree_edges)
    incidence_matrix = np.zeros((num_edges, N))

    for idx, (i, j) in enumerate(tree_edges):
        incidence_matrix[idx, i] = -1 
        incidence_matrix[idx, j] = 1 

    A = scipy.sparse.kron(incidence_matrix, scipy.sparse.eye(*qq))

    return incidence_matrix, A


def CreateA(oriZ,
            ini_Esttheta_or_location=None,
            way='Full',
            KNN_num=2,
            ini_graph=None,
            seed=None):
    N, qq = oriZ.shape
    qq = (qq, qq)

    if way == 'MST_r' and ini_graph is None:
        if seed is not None:
            np.random.seed(seed)
        else:
            seed = int(time())
            np.random.seed(seed)
        points = np.random.rand(N, 2)
        inc_matrix, A = mst_incidence_matrix(points, qq)
        return inc_matrix, A

    elif way == 'MST_r' and ini_graph is not None:
        # ini_graph 是原始图的邻接矩阵
        seed = int(time())
        np.random.seed(seed)
        inc_matrix, A = random_spanning_tree_incidence_matrix(ini_graph, qq)

        return inc_matrix, A

    elif way == 'MST_i' and ini_Esttheta_or_location is not None:
        if ini_graph is not None:
            inc_matrix, A = get_mst_adjacency_matrix(ini_Esttheta_or_location,
                                                     ini_graph, qq)
        else:
            inc_matrix, A = mst_incidence_matrix(ini_Esttheta_or_location, qq)
        return inc_matrix, A

    elif way == 'KNN' and ini_Esttheta_or_location is not None:
        inc_matrix, A = knn_graph_incidence_matrix(ini_Esttheta_or_location,
                                                   KNN_num, qq)
        return inc_matrix, A

    elif way == 'Full':
        inc_matrix, A = wm.createA(oriZ)
        return inc_matrix, A

    elif way == 'Forest':
        pass

    else:
        print('error')
