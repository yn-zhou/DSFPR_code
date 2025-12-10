import numpy as np
'''
为模拟生成数据集
'''


def gen_center(hete_dim, Knum):
    start_num = 1
    array = np.arange(start_num, start_num + 2 * Knum, 2)
    center = np.tile(array, (hete_dim, 1)).T
    return center


def expand_list(input_list, n):
    """
    将一个 K 维列表扩充为 Kn 维列，其中每个元素都重复 n 次
    """
    expanded_list = np.repeat(input_list, n)
    return expanded_list


def gen_data(label_element,
             n,
             qq,
             pp,
             is_hete_intercept=True,
             gen_way='uniform',
             seed=42):
    '''
    2个组
    2台local机器
    qq:3维异质性数据,包含截距项
    pp:2维同质性数据
    gen_way:生成数据的方式，'normal'为正态分布，'uniform'为均匀分布, 'multinomial'为多项分布
    '''

    np.random.seed(seed)

    real_label = expand_list(label_element, n)
    sample_size = len(real_label)

    real_X = np.random.uniform(0, 1, size=(sample_size, pp))

    if gen_way == 'test':
        # 维度
        d = pp  # 可修改为任意维度

        # 构造协方差矩阵
        cov_matrix = np.full((d, d), 0.3)  # 先填充0.3
        np.fill_diagonal(cov_matrix, 1)  # 对角线设为1

        # 均值向量（假设均值为0）
        mean_vector = np.zeros(d)

        # 生成样本
        real_X = np.random.multivariate_normal(mean_vector,
                                               cov_matrix,
                                               size=sample_size)

        # real_Z = np.random.uniform(0, 1, size=(sample_size, qq))
        n_trials = 1  # 每次实验的试验次数
        p_success = 0.5  # 成功的概率
        # 生成一个 N*qq 的矩阵，每个元素随机来自于一个二项分布
        real_Z = np.random.binomial(n_trials,
                                    p_success,
                                    size=(sample_size, qq))

    if gen_way == 'uniform':
        real_Z = np.random.uniform(0, 1, size=(sample_size, qq))
    if gen_way == 'normal':
        real_Z = np.random.normal(0, 1, size=(sample_size, qq))
    if gen_way == 'multinomial':
        # 定义多项分布的概率
        probabilities = [0.2, 0.3, 0.1, 0.25, 0.15]
        # 生成一个 N*k 的矩阵，每个元素随机来自于一个多项分布，热编码的形式
        real_Z = np.random.multinomial(1,
                                       probabilities,
                                       size=(sample_size, qq))
    if gen_way == 'binomial':
        # 定义二项分布的参数
        n_trials = 1  # 每次实验的试验次数
        p_success = 0.5  # 成功的概率
        # 生成一个 N*qq 的矩阵，每个元素随机来自于一个二项分布
        real_Z = np.random.binomial(n_trials,
                                    p_success,
                                    size=(sample_size, qq))

    if gen_way == 'binomial_uniform_mixed':
        # 维度
        d = pp  # 可修改为任意维度

        # 构造协方差矩阵
        cov_matrix = np.full((d, d), 0.3)  # 先填充0.3
        np.fill_diagonal(cov_matrix, 1)  # 对角线设为1

        # 均值向量（假设均值为0）
        mean_vector = np.zeros(d)

        # 生成样本
        real_X = np.random.multivariate_normal(mean_vector,
                                               cov_matrix,
                                               size=sample_size)
        qq1 = qq // 2
        qq2 = qq - qq1

        # 定义二项分布的参数
        n_trials = 1  # 每次实验的试验次数
        p_success = 0.5  # 成功的概率
        real_Z1 = np.random.binomial(n_trials,
                                     p_success,
                                     size=(sample_size, qq2))

        real_Z2 = np.random.uniform(0, 1, size=(sample_size, qq1))

        real_Z = np.hstack([real_Z1, real_Z2])

    if gen_way == 'binomial_normal_mixed':
        # 维度
        d = pp  # 可修改为任意维度

        # 构造协方差矩阵
        cov_matrix = np.full((d, d), 0)  # 先填充0.3
        np.fill_diagonal(cov_matrix, 1)  # 对角线设为1

        # 均值向量（假设均值为0）
        mean_vector = np.zeros(d)

        # 生成样本
        real_X = np.random.multivariate_normal(mean_vector,
                                               cov_matrix,
                                               size=sample_size)

        qq1 = 1
        qq2 = qq - qq1

        # 定义二项分布的参数
        n_trials = 1  # 每次实验的试验次数
        p_success = 0.5  # 成功的概率
        real_Z1 = np.random.binomial(n_trials,
                                     p_success,
                                     size=(sample_size, qq1))
        # real_Z1 = np.zeros((sample_size, qq2))
        # for i in range(sample_size):
        #     if (real_X[i][0])**2 > 1:
        #         real_Z1[i] = 1
        #     else:
        #         real_Z1[i] = 0

        real_Z2 = np.random.uniform(0, 1, size=(sample_size, qq2))
        # real_Z2 = np.random.multivariate_normal(np.zeros(int(qq2)),
        #                                         np.eye(int(qq2)), sample_size)

        real_Z = np.hstack([real_Z1, real_Z2])

    if gen_way == 'normal_uniform_mixed':
        # 维度
        d = pp  # 可修改为任意维度

        # 构造协方差矩阵
        cov_matrix = np.full((d, d), 0)  # 先填充0.3
        np.fill_diagonal(cov_matrix, 1)  # 对角线设为1

        # 均值向量（假设均值为0）
        mean_vector = np.zeros(d)

        # 生成样本
        real_X = np.random.multivariate_normal(mean_vector,
                                               cov_matrix,
                                               size=sample_size)

        real_Z = np.random.uniform(0, 1, size=(sample_size, qq))

    if is_hete_intercept:
        oriZ = np.hstack([np.ones((sample_size, 1)), real_Z])
        oriX = real_X
    else:
        oriZ = real_Z
        oriX = np.hstack([np.ones((sample_size, 1)), real_X])

    real_beta = np.ones(oriX.shape[1])

    center = gen_center(oriZ.shape[1], len(label_element))

    oriY = np.zeros(sample_size)
    for i in range(sample_size):
        oriY[i] = np.dot(center[real_label[i]], oriZ[i]) + np.dot(
            real_beta, oriX[i]) + np.random.normal(0, 0.3)

    indicator_matrix = np.eye(len(label_element))[real_label]
    real_Theta = np.dot(indicator_matrix, center[:len(label_element)])

    return oriY, real_X, real_Z, real_label, real_beta, real_Theta


def gen_distributed_data(label_element,
                         n,
                         qq,
                         pp,
                         is_hete_intercept=True,
                         machine_num=2,
                         gen_way='uniform',
                         seed=42):

    np.random.seed(seed)

    oriY, real_X, real_Z, real_label, real_beta, real_Theta = gen_data(
        label_element=label_element,
        n=n,
        qq=qq,
        pp=pp,
        is_hete_intercept=is_hete_intercept,
        gen_way=gen_way,
        seed=seed)

    sample_size = len(real_label)
    numbers = np.arange(sample_size)
    # 打乱数字顺序
    np.random.shuffle(numbers)
    distributed_index = np.array_split(numbers, machine_num)

    local_real_Y = []
    local_real_X = []
    local_real_Z = []
    local_real_label = []
    local_real_Theta = []
    for i in range(machine_num):
        local_real_Y.append(oriY[distributed_index[i]])
        local_real_X.append(real_X[distributed_index[i]])
        local_real_Z.append(real_Z[distributed_index[i]])
        local_real_label.append(real_label[distributed_index[i]])
        local_real_Theta.append(real_Theta[distributed_index[i]])

    final_data = {
        # 'oriY': oriY,
        # 'real_X': real_X,
        # 'real_Z': real_Z,
        # 'real_label': real_label,
        # 'real_Theta': real_Theta,
        'real_beta': real_beta,
        'oriY_shuffle': oriY[numbers],
        'real_X_shuffle': real_X[numbers],
        'real_Z_shuffle': real_Z[numbers],
        'real_label_shuffle': real_label[numbers],
        'real_Theta_shuffle': real_Theta[numbers],
        # 'sample_index': numbers,
        'distributed_index': distributed_index
    }

    return local_real_Y, local_real_X, local_real_Z, local_real_label, real_beta, local_real_Theta, final_data
