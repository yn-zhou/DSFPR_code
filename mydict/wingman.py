import numpy as np
from networkx.algorithms.components.connected import connected_components
from sklearn import preprocessing
import scipy, math, itertools, multiprocessing, time, networkx, copy
from scipy.spatial.distance import pdist
import statsmodels.api as sm
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix, csc_matrix, save_npz, load_npz
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, rand_score, silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.linear_model import LinearRegression
# from dict.linear_regression_mixtures import LinearRegressionsMixture
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt



def createLAM(nested_lists, H):
    Lam0 = list(
        itertools.product(nested_lists, repeat=H)
    ) 
    return Lam0


def createX(n, mu, cov):  
    Z = np.random.multivariate_normal(mu, cov, n, check_valid='raise')
    x_scaled = preprocessing.scale(Z)  
    return x_scaled


def createY(oriZ, oriX, beta, weights, theta, stddev):
    N, pp = oriX.shape
    qq = oriZ.shape[1]
    weight = np.insert(np.cumsum(weights), 0, 0)
    random = np.random.rand(N)
    place = [0] * (len(weight) - 1)
    realtheta = np.zeros((N, qq))
    oriY = np.zeros((N, 2))
    oriY[:, 0] = np.dot(oriX, beta)
    for i in range(len(weight) - 1):
        place[i] = np.where((random < weight[i + 1]) & (random >= weight[i]))
        realtheta[place[i]] = theta[i]
        oriY[:, 1][place[i]] = int(i)
        oriY[:, 0][place[i]] = oriY[:, 0][place[i]] + np.dot(
            oriZ[place[i]], theta[i]) + np.random.normal(
                0, stddev[i], len(place[i]))
    return oriY, realtheta


def createY_1(oriZ, oriX, beta, weights, theta, stddev):  
    N, pp = oriX.shape
    qq = oriZ.shape[1]
    weight = np.insert(np.cumsum(weights), 0, 0)
    random = np.random.rand(N)
    place = [0] * (len(weight) - 1)
    realtheta = np.zeros((N, qq))
    oriY = np.zeros((N, 2))
    oriY[:, 0] = np.dot(oriX, beta)
    for i in range(len(weight) - 1):
        place[i] = np.where((random < weight[i + 1]) & (random >= weight[i]))
        realtheta[place[i]] = theta[i]
        oriY[:, 1][place[i]] = int(i)
        oriY[:, 0][place[i]] = oriY[:, 0][place[i]] + np.dot(
            oriZ[place[i]], theta[i]) + np.random.normal(
                0, stddev[i], len(place[i]))
    return oriY, realtheta




def createY_fixed_proportion(oriZ, oriX, beta, weights, theta, stddev):
    N, pp = oriX.shape
    qq = oriZ.shape[1]
    weight = np.insert(np.cumsum(weights), 0, 0) * N
    num = np.arange(N)
    place = [0] * (len(weight) - 1)
    realtheta = np.zeros((N, qq))
    oriY = np.zeros((N, 2))
    oriY[:, 0] = np.dot(oriX, beta)
    for i in range(len(weight) - 1):
        place[i] = np.where((num < weight[i + 1]) & (num >= weight[i]))
        realtheta[place[i]] = theta[i]
        oriY[:, 1][place[i]] = int(i)
        oriY[:, 0][place[i]] = oriY[:, 0][place[i]] + np.dot(
            oriZ[place[i]], theta[i]) + np.random.normal(
                0, stddev[i], len(place[i]))
    return oriY, realtheta


def ridge(oriZ, oriX, oriY, lamstar=0.001):

    Z = block_diag_matrix(oriZ)
    N, qq = oriZ.shape
    ATA = createATA(oriZ)

    XTX_1 = np.linalg.inv(np.dot(oriX.T, oriX))

    XTX_1XT = np.dot(XTX_1, oriX.T)
    Q_X = np.identity(N) - np.dot(oriX, XTX_1XT)
    ZTQ_XZ = np.dot(np.dot(Z.T, Q_X), Z)
    M1 = np.linalg.inv(ZTQ_XZ + lamstar * ATA)
    M2 = np.dot(np.dot(Z.T, Q_X), oriY[:, 0])

    esttheta = np.dot(M1, M2)
    estbeta = np.dot(XTX_1XT, oriY[:, 0] - np.dot(Z, esttheta))
    return esttheta, estbeta


def Iniest(oriZ, oriX, oriY, lamstar=0.001, way='none', Inik='none'):
    if len(oriY.shape) == 1:
        oriY = np.vstack((oriY, np.ones(len(oriY)))).T
    if way == 'knn':
        esttheta, estbeta = find_nearest_k_samples_and_return_estimate(
            oriZ, oriX, oriY, Inik)
    else:
        esttheta, estbeta = ridge(oriZ, oriX, oriY, lamstar)

    if way == 'kmeans':
        Cluster_objects = esttheta.reshape(oriZ.shape)
        esttheta, estbeta = kmeans_and_least_squares_fit(
            Cluster_objects, oriZ, oriX, oriY, Inik)
    return esttheta, estbeta


def Iniest_oracle(oriZ,
                  oriX,
                  oriY,
                  labels,
                  lamstar=0.001,
                  way='none',
                  Inik='none'):
    oracle_beta, oracle_Zeta, oracle_theta = oracle(oriZ, oriX, oriY, labels)

    return oracle_theta.ravel(), oracle_beta


def paraller_Iniest(spZ, spX, spY, lamstar=0.001, way='none', Inik='none'):
    H = len(spX)
    esttheta = [0] * H
    estbeta = [0] * H
    for h in range(H):
        esttheta[h], estbeta[h] = Iniest(spZ[h], spX[h], spY[h], lamstar, way,
                                         Inik)
    return esttheta, estbeta


def kmeans_and_least_squares_fit(Cluster_objects,
                                 oriZ,
                                 oriX,
                                 oriY,
                                 Inik='none'):
    N, qq = oriZ.shape
    pp = oriX.shape[1]
    if Inik == 'none':
        inik = math.floor(np.sqrt(N))
        if inik <= 5:
            inik = 5
    else:
        inik = Inik
    kmeans = KMeans(n_clusters=inik, random_state=0,
                    n_init="auto").fit(Cluster_objects)
    estlabel = kmeans.labels_
    Coefficients = least_squares_fit_group(oriZ, oriX, oriY, estlabel)

    coefficients = np.zeros((N, pp + qq))
    for i in range(N):
        coefficients[i] = Coefficients[estlabel[i]]

    return coefficients[:, :qq].ravel(), np.mean(coefficients[:, qq:], axis=0)


def find_nearest_k_samples_and_return_estimate(oriZ, oriX, oriY, Inik='none'):

    N, pp = oriX.shape
    qq = oriZ.shape[1]
    if Inik == 'none':
        inik = math.floor(np.sqrt(N))
        if inik <= 5:
            inik = 5
    else:
        inik = Inik

    data = np.hstack((oriZ, oriX, oriY[:, 0].reshape(-1, 1)))
    nn_model = NearestNeighbors(n_neighbors=inik, algorithm='kd_tree')
    nn_model.fit(data)

    distances, indices = nn_model.kneighbors(data)
    nearest_samples = data[indices]
    coefficients = np.zeros((data.shape[0], data.shape[1] - 1))
    for i in range(data.shape[0]):
        X = nearest_samples[i][:, :-1]  
        y = nearest_samples[i][:, -1]  
        coefficients[i] = model.params
    return coefficients[:, :qq].ravel(), np.mean(coefficients[:, qq:], axis=0)


def least_squares_fit(oriZ, oriX, oriY):
    if oriX is None:
        data = np.hstack((oriZ, oriX, oriY))
    else:
        data = np.hstack((oriZ, oriY))
    X = data[:, :-1]  
    y = data[:, -1]  
    model = sm.OLS(y, X).fit()
    coef = model.params

    return coef[:oriZ.shape[1]]


def least_squares_fit_group(oriZ, oriX, oriY, estlabel):

    if oriX is None:
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



#返回各种个体的估计值
def least_squares_fit_group_return(oriZ, oriX, oriY, estlabel):
    qq = oriZ.shape[1]
    estlabel = estlabel.astype(int)
    coefficients = least_squares_fit_group(oriZ, oriX, oriY, estlabel)
    Esttheta = np.zeros((len(estlabel), oriZ.shape[1]))
    for i in range(len(estlabel)):
        Esttheta[i] = coefficients[estlabel[i]][:qq]
    return Esttheta, coefficients[:, :qq], np.mean(coefficients[:, qq:],
                                                   axis=0)


def quasi_oracle(oriZ, oriX, oriY, real_labels, est_labels):
    real_labels = np.array(real_labels).astype(int)
    est_labels = np.array(est_labels).astype(int)

    N, qq = oriZ.shape
    quasi_labels = np.zeros(N)
    real_knum = len(np.unique(real_labels))
    est_knum = len(np.unique(est_labels))

    for k in range(est_knum):
        location_k = np.where(est_labels == k)[0]
        the_k = np.bincount((real_labels[location_k]).astype(int)).argmax()
        quasi_labels[location_k] = the_k

    quasi_RI = rand_score(real_labels, quasi_labels)
    quasi_Cm = pair_confusion_matrix(real_labels, quasi_labels)
    quasi_FDR = quasi_Cm[0, 1] / (quasi_Cm[0, 1] + quasi_Cm[1, 1])
    quasi_TRP = quasi_Cm[1, 1] / (quasi_Cm[1, 0] + quasi_Cm[1, 1])
    return oracle(oriZ, oriX, oriY,
                  quasi_labels), (quasi_RI, quasi_FDR, quasi_TRP, quasi_Cm)


def return_index(real_labels, est_labels):
    RI = rand_score(real_labels, est_labels)
    Cm = pair_confusion_matrix(real_labels, est_labels)
    FDR = Cm[0, 1] / (Cm[0, 1] + Cm[1, 1])
    TRP = Cm[1, 1] / (Cm[1, 0] + Cm[1, 1])
    return RI, FDR, TRP, Cm


def oracle(oriZ, oriX, oriY, Label):
    pp = oriX.shape[1]
    qq = oriZ.shape[1]
    Label = np.array(Label).astype(int)
    realknum = len(np.unique(Label))

    Z = block_diag_matrix(oriZ)

    indicate_matrix = np.kron(np.eye(realknum)[Label], np.identity(qq))
    Z_DX = np.hstack((Z.dot(indicate_matrix), oriX))
    invZ_DX = np.linalg.inv(np.dot(Z_DX.T, Z_DX))

    params = np.dot(np.dot(invZ_DX, Z_DX.T), oriY)

    oracle_beta = params[-pp:]

    oracle_zeta = params[:-pp]
    oracle_Zeta = oracle_zeta.reshape((realknum, qq))
    oracle_Theta = np.zeros((len(oriY), qq))
    for i in range(len(oriY)):
        oracle_Theta[i] = oracle_Zeta[Label[i]]

    return oracle_beta, oracle_Zeta, oracle_Theta


def Oracle(oriZ, oriX, oriY, estlabel):
    qq = oriZ.shape[1]
    pp = oriX.shape[1]
    estlabel = estlabel.astype(int)
    coefficients = least_squares_fit_group(oriZ, oriX, oriY, estlabel)
    Estcoeff = np.zeros((len(estlabel), pp + qq))
    for i in range(len(estlabel)):
        Estcoeff[i] = coefficients[estlabel[i]]
    return Estcoeff, coefficients


def ST(z, t):
    res = np.zeros(z.shape)
    z_norm = np.linalg.norm(z, ord=2, axis=1)
    # np.seterr(divide='ignore', invalid='ignore')
    place = z_norm - t
    bb = np.where(place > 0)

    if isinstance(t, (int, float)):
        aa = 1 - (t / z_norm[bb])
    else:
        aa = 1 - (t[bb] / z_norm[bb])
    res[bb] = aa.reshape(-1, 1) * z[bb]
    return res


def MCP(lastxi, gamma, Lam, vertheta):
    res = np.zeros(lastxi.shape)
    lastxi_norm = np.linalg.norm(lastxi, ord=2, axis=1)

    wh1 = np.where(lastxi_norm <= gamma * Lam)
    wh2 = np.where(lastxi_norm > gamma * Lam)
    cons = 1 - 1 / (gamma * vertheta)
    res[wh1] = ST(lastxi[wh1], Lam / vertheta) / cons
    res[wh2] = lastxi[wh2]
    return res




def MCP_weighted(xi, gamma, Lam, vertheta, Sm, weight_way='original'):  #加权MCP



    if weight_way == 'ward_norm':
        res = (Sm[:, np.newaxis]) * xi
        Sm_xi = (Sm[:, np.newaxis]) * xi
    else:
        res = xi.copy()



    xi_norm = np.linalg.norm(res, ord=2, axis=1)

    wh1 = np.where(xi_norm <= (gamma * Lam))
    if weight_way == 'ward_norm':

        cons = (gamma * vertheta) / (gamma * vertheta - 1)
        res[wh1] = ST(Sm_xi[wh1], Lam / vertheta) * cons
    else:
        res[wh1] = (
            (gamma * vertheta) * ST(xi[wh1], (Lam / vertheta) * Sm[wh1])) / (
                (gamma * vertheta - Sm[wh1])[:, None])


    if weight_way == 'ward_norm':
        res = res / (Sm[:, np.newaxis])



    return res




def incidence_matrix(N):

    G = networkx.complete_graph(N)
    incidence_matrix = networkx.incidence_matrix(
        G, oriented=True).transpose().toarray()

    return incidence_matrix


def createA(oriZ):

    N, qq = oriZ.shape
    qq = (qq, qq)
    G = networkx.complete_graph(N)
    incidence_matrix = networkx.incidence_matrix(
        G, oriented=True).transpose().tocsc()
    A = scipy.sparse.kron(incidence_matrix, scipy.sparse.eye(*qq))
    return incidence_matrix, A


def createATA(oriZ):
    N, qq = oriZ.shape
    a = np.kron(np.ones(N), np.identity(qq)).T
    return N * np.identity(N * qq) - np.dot(a, a.T)


def dir_createATA(A):
    AT = A.transpose()
    return AT.dot(A)


def paraller_createAn(spZ):  
    start_time = time.time()
    H = len(spZ)
    n = [0] * H
    ATA = [0] * H

    for h in range(H):
        n[h] = len(spZ[h])
        ATA[h] = createATA(spZ[h])
    if len(set(n)) == 1:
        an = createA(spZ[0])[1]
        anT = an.transpose()
        An = [an] * H
        AnT = [anT] * H
    else:
        An = [0] * H
        AnT = [0] * H
        for h in range(H):
            An[h] = createA(spZ[h])[1]
            AnT[h] = An[h].transpose()

    end_time = time.time()
    run_time = end_time - start_time
    print("createAn runing time：", run_time)
    return n, ATA, An, AnT


def VtoM(V):
    n = int((1 + np.sqrt(1 + 8 * len(V))) / 2)  
    matrix = np.ones((n, n)) * 100
    matrix[0] = np.concatenate((np.ones(1) * 100, np.abs(V[0:n - 1])))
    for i in range(1, n - 1):
        num1 = int((2 * n - i - 1) * (i) / 2)
        num2 = int((2 * n - i - 2) * (i + 1) / 2)
        matrix[i] = np.concatenate(
            (np.ones(i + 1) * 100, np.abs(V[num1:num2])))
    for j in range(n):
        matrix[j, j] = 0
    return matrix


def merge(l):

    def to_graph(l):
        G = networkx.Graph()
        for part in l:
            G.add_nodes_from(part)
            G.add_edges_from(to_edges(part))
        return G

    def to_edges(l):
        """
            treat `l` as a Graph and returns it's edges
            to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
        """
        it = iter(l)
        last = next(it)

        for current in it:
            yield last, current
            last = current
    
    Graph = to_graph(l)
    return list(connected_components(Graph))



def Group(etamatrix, doorsill=0.01):
    N = etamatrix.shape[0]
    etamatrix_no_nan = np.nan_to_num(etamatrix, nan=np.inf)
    crood = np.array(np.where(etamatrix_no_nan <= doorsill)).T
    crood = np.vstack((np.array([np.arange(0, N), np.arange(0, N)]).T, crood))
    G = merge(crood)
    return G


# 第二种做法为在迭代过程中，若eta小于某个阈值就聚为一类？


def toZ(etamatrix, n, doorsill=0.01):
    # n=len(etamatrix)
    G = Group(np.abs(etamatrix), doorsill)
    # print('G',G)
    label = np.arange(n)
    for k in range(len(G)):
        label[np.array(list(G[k]))] = k
    label = label.astype(int)
    indicator_matrix = np.eye(len(G))[label]
    return indicator_matrix


def block_diag_matrix(oriZ):  #将oriX写出对角形式diag{x_1,x_2,...,x_n}
    input_array = oriZ

    num_rows, num_cols = input_array.shape

    # 计算分块对角矩阵的总大小
    block_diag_rows = num_rows
    block_diag_cols = num_rows * num_cols
    # 创建一个空的分块对角矩阵
    block_diag_matrix = np.zeros((block_diag_rows, block_diag_cols))

    # 填充分块对角矩阵的各个块
    for i in range(num_rows):
        block_diag_matrix[i, i * num_cols:(i + 1) * num_cols] = input_array[i]

    return sp.csr_matrix(block_diag_matrix)


def creatSM(H, qq, grouph, knum1):  
    knum1 = np.array(knum1).astype(int)
    SMh = [0] * H
    SM = []

    for h in range(len(grouph)):
        SMh[h] = [len(grouph[h][0])]

        SM.append(len(grouph[h][0]))
        if len(grouph[h]) > 1:
            for j in range(1, len(grouph[h])):
                SM.append(len(grouph[h][j])) 
                SMh[h].append(len(grouph[h][j]))  

    Smself = [0] * H

    Smdif = [[0 for i in range(H)] for j in range(H)]
    for h in range(H):
        midv = np.triu(np.outer(SMh[h], SMh[h]), 1)
        Smself[h] = midv[np.nonzero(midv)]
        for m in range(h + 1, H):
            # print([h,m])
            Smdif[h][m] = np.outer(SMh[h], SMh[m]).ravel()  

    ss = Smdif[0][1]  
    for h in range(H):
        for m in range(2, H):
            if isinstance(Smdif[h][m], np.ndarray) is True: 
                ss = np.block([ss, Smdif[h][m]])

    indc = incidence_matrix(H).tolist()  
    Astar = [[0 for i in range(H)] for j in range(len(indc))]
    for i in range(len(indc)):
        a = indc[i].index(1.0)
        b = indc[i].index(-1.0)
        Astar[i][a] = np.kron(np.identity(knum1[a]), np.ones(knum1[b])).T
        Astar[i][b] = (-1) * np.kron(np.ones(knum1[a]), np.identity(
            knum1[b])).T
        aset = [a, b]
        set = np.delete(np.arange(H), aset)
        for j in set:
            Astar[i][j] = np.array([0])

    shape = [[0 for i in range(H)] for j in range(len(indc))]

    for i in range(len(indc)):
        for h in range(H):
            Z = len(Astar[i][h].shape)
            if Z >= 2:
                shape[i][h] = np.array(Astar[i][h].shape)
            else:
                shape[i][h] = np.array([0, 0])

    mx = [0] * len(indc)
    my = [0] * H
    reshape = [[0 for i in range(H)] for j in range(len(indc))]

    for i in range(len(indc)):
        for h in range(H):
            if shape[i][h][0] > mx[i]:
                mx[i] = shape[i][h][0]
            if shape[i][h][1] > my[h]:
                my[h] = shape[i][h][1]

    for i in range(len(indc)):
        for h in range(H):
            reshape[i][h] = [mx[i], my[h]]

    AAstar = Astar[0][0]
    for h in range(1, H):
        if (shape[0][h] == np.array([0, 0])).all():
            AAstar = np.block([AAstar, np.zeros(reshape[0][h])])
        else:
            AAstar = np.block([AAstar, Astar[0][h]])

    for i in range(1, len(indc)):
        if (shape[i][0] == np.array([0, 0])).all():
            BB = np.zeros(reshape[i][0])
        else:
            BB = Astar[i][0]
        for h in range(1, H):
            if (shape[i][h] == np.array([0, 0])).all():
                BB = np.block([BB, np.zeros(reshape[i][h])])
            else:
                BB = np.block([BB, Astar[i][h]])
        AAstar = np.block([[AAstar], [BB]])

    AAstar = csc_matrix(AAstar)
    qq = (qq, qq)
    KAAstar = scipy.sparse.kron(AAstar, scipy.sparse.eye(*qq))

    return KAAstar, Smself, ss


def create_weight(H, qq, grouph, knum1, weight_way='original'):

    knum1 = knum1.astype(int)
    SMh = [0] * H
    SM = []

    for h in range(len(grouph)):
        SMh[h] = [len(grouph[h][0])]

        SM.append(len(grouph[h][0]))
        if len(grouph[h]) > 1:
            for j in range(1, len(grouph[h])):

                SMh[h].append(len(grouph[h][j])) 
                SM.append(len(grouph[h][j])) 
        SMh[h] = np.array(SMh[h])

    SM = np.array(SM)

    Smself = [0] * H

    Smdif = [[0 for i in range(H)] for j in range(H)]
    for h in range(H):
        if weight_way == 'ward_weight':
            midv = np.add.outer(1 / SMh[h], 1 / SMh[h])
        elif weight_way == 'ward_norm':
            midv = np.sqrt(1 / (np.add.outer(1 / SMh[h], 1 / SMh[h])))
        elif weight_way == 'original':
            midv = np.outer(SMh[h], SMh[h])

        mmidv = np.triu(midv, 1)
        Smself[h] = mmidv[np.nonzero(mmidv)]
        for m in range(h + 1, H):
            if weight_way == 'ward_weight':
                midv = np.add.outer(1 / SMh[h], 1 / SMh[m])
            elif weight_way == 'ward_norm':
                midv = np.sqrt(1 / (np.add.outer(1 / SMh[h], 1 / SMh[m])))
            else:
                midv = np.outer(SMh[h], SMh[m])
            Smdif[h][m] = midv.ravel()  

    ss = Smdif[0][1] 
    for h in range(H):
        for m in range(2, H):
            if isinstance(Smdif[h][m], np.ndarray) is True:  
                ss = np.block([ss, Smdif[h][m]])

    indc = incidence_matrix(H).tolist()  
    Astar = [[0 for i in range(H)] for j in range(len(indc))]
    for i in range(len(indc)):
        a = indc[i].index(1.0)
        b = indc[i].index(-1.0)

        Astar[i][a] = np.kron(np.identity(knum1[a]), np.ones(knum1[b])).T
        Astar[i][b] = (-1) * np.kron(np.ones(knum1[a]), np.identity(
            knum1[b])).T
        aset = [a, b]
        set = np.delete(np.arange(H), aset)
        for j in set:
            Astar[i][j] = np.array([0])

    shape = [[0 for i in range(H)] for j in range(len(indc))]

    for i in range(len(indc)):
        for h in range(H):
            Z = len(Astar[i][h].shape)
            if Z >= 2:
                shape[i][h] = np.array(Astar[i][h].shape)
            else:
                shape[i][h] = np.array([0, 0])

    mx = [0] * len(indc)
    my = [0] * H
    reshape = [[0 for i in range(H)] for j in range(len(indc))]

    for i in range(len(indc)):
        for h in range(H):
            if shape[i][h][0] > mx[i]:
                mx[i] = shape[i][h][0]
            if shape[i][h][1] > my[h]:
                my[h] = shape[i][h][1]

    for i in range(len(indc)):
        for h in range(H):
            reshape[i][h] = [mx[i], my[h]]

    AAstar = Astar[0][0]
    for h in range(1, H):
        if (shape[0][h] == np.array([0, 0])).all():
            AAstar = np.block([AAstar, np.zeros(reshape[0][h])])
        else:
            AAstar = np.block([AAstar, Astar[0][h]])

    for i in range(1, len(indc)):
        if (shape[i][0] == np.array([0, 0])).all():
            BB = np.zeros(reshape[i][0])
        else:
            BB = Astar[i][0]
        for h in range(1, H):
            if (shape[i][h] == np.array([0, 0])).all():
                BB = np.block([BB, np.zeros(reshape[i][h])])
            else:
                BB = np.block([BB, Astar[i][h]])
        AAstar = np.block([[AAstar], [BB]])

    AAstar = csc_matrix(AAstar)
    qq = (qq, qq)
    KAAstar = scipy.sparse.kron(AAstar, scipy.sparse.eye(*qq))

    return KAAstar, Smself, ss


def tiqu(A):

    tril_mask = np.triu(np.ones_like(A, dtype=bool), k=1)
    new_vector = np.take(A.ravel(), np.where(tril_mask.ravel() is True))[0]

    return new_vector


def subplots_liner_chart(lamlist, BIC, RI, FDR, Knum, best_lambda, realKnum):

    fig, axs = plt.subplots(4, 1, figsize=(6, 8))

    axs[0].plot(lamlist, BIC)
    # axs[0].set_title('BIC')
    axs[0].set_ylabel('BIC')
    axs[0].axvline(x=best_lambda, color='red', linestyle='--')  

    axs[1].plot(lamlist, RI)
    # axs[2].set_title('RI')
    axs[1].set_ylabel('RI')
    axs[1].axvline(x=best_lambda, color='red', linestyle='--')  

    axs[2].plot(lamlist, FDR)
    # axs[3].set_title('FDR')
    axs[2].set_ylabel('FDR')
    axs[2].axvline(x=best_lambda, color='red', linestyle='--') 
    axs[3].plot(lamlist, Knum)
    # axs[1].set_title('Knum')
    axs[3].set_ylabel('Knum')
    axs[3].set_xlabel('$\lambda$')
    axs[3].axhline(y=realKnum, color='#808080', linestyle='--')  
    axs[3].axvline(x=best_lambda, color='red', linestyle='--')  

    # 调整子图之间的间距
    plt.tight_layout()
    plt.show()


def plots_liner_chart(lamlist, BIC, RI, FDR, Knum, best_lambda, realKnum):

    BIC = (BIC - np.min(BIC)) / (np.max(BIC) - np.min(BIC))


    fig = plt.figure()
    ax = fig.add_subplot(111)

    lns1 = ax.plot(lamlist, BIC, '-', label='BIC', color='C3')
    lns2 = ax.plot(lamlist, RI, '-', label='RI', color='C0')
    lns3 = ax.plot(lamlist, FDR, '-', label='FDR', color='C1')

    ax2 = ax.twinx()
    lns4 = ax2.plot(lamlist, Knum, '-', label='Knum', color='C2')

    # added these three lines
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    ax.grid()
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel(r"Normalized quantity")
    ax2.set_ylabel(r"Estimate K")

    ax.axvline(x=best_lambda, color='#808080', linestyle='--')  
    ax2.axhline(y=realKnum, color='#808080', linestyle='--')  
    ax.set_xlim(min(lamlist), max(lamlist))
    ax.set_ylim(-0.1, 1.1)
    ax2.set_ylim(0, 10)

    plt.show()
