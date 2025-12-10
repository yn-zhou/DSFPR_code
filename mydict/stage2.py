import numpy as np
import main_sa as sa
import stage1 as st1
import wingman as wm
from copy import copy
from time import time
import multiprocessing
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



class master_machine:

    def __init__(self,
                 machine_list: list,
                 oriZ,
                 oriY,
                 oriX,
                 lam_list_st2=[1, 5, 100],
                 real_label=None,
                 real_beta=None,
                 real_theta=None,
                 vertheta=3,
                 gamma=1,
                 doorsill=0.01,
                 tolerance=0.001,
                 T0=50,
                 T1=100,
                 c1=0.5,
                 get_res=1,
                 hot_start=False):

        self.machine_list = machine_list
        if not all(isinstance(a, st1.local_machine) for a in machine_list):
            raise ValueError(
                "All elements must be instances of machine(class)")


        self.pp = self.machine_list[0].pp  
        self.qq = self.machine_list[0].qq
        for a in self.machine_list[1:]:  
            if a.pp != self.pp:
                raise ValueError(
                    f"ClassA instance '{a.machine_index}' has a different pp value: {a.pp}!"
                )
            if a.qq != self.qq:
                raise ValueError(
                    f"ClassA instance '{a.machine_index}' has a different self.qq value: {a.qq}!"
                )

        self.machine_num = len(machine_list)
        self.each_sample_size = [] * self.machine_num
        for a in self.machine_list:
            self.each_sample_size.append(a.oriY.shape[0])
        self.all_sample_size = sum(self.each_sample_size)
        self.N = self.all_sample_size

        self.lam_list_st2 = lam_list_st2
        self.oriZ = oriZ
        self.oriY = oriY
        self.oriX = oriX

        self.real_label = real_label
        self.real_beta = real_beta
        self.real_theta = real_theta
        self.vertheta = vertheta
        self.gamma = gamma
        self.doorsill = doorsill
        self.tolerance = tolerance
        self.T0 = T0
        self.T1 = T1
        self.c2 = c1
        self.get_res2 = get_res
        self.hot_start = hot_start
        self.train_switch_st1 = False

        self.all_train_switch = False

        self.ini_process()

    def ini_process(self):
        H = self.machine_num

        self.ini_local_Alpha = [None] * H
        self.ini_local_vec_Alpha = [None] * H
        self.ini_local_Knum = [None] * H

        self.local_samplelabel = [None] * H

        self.indicate_matrix = [None] * H
        self.ini_estbeta = [None] * H
        self.local_ini_EstTheta = [None] * H

        self.AmTAm = [None] * H
        self.Am = [None] * H
        self.AmT = [None] * H

        self.grouph = [None] * H
        self.estbeta = [None] * H

        self.ini_sigma = [None] * H
        self.ini_vec_sigma = [None] * H

        self.samplelabel_stage1 = []
        self.Knum_stage1 = 0
        self.kforeachgroup_stage1 = []
        self.local_kforeachgroup_stage1 = []

        self.local_RI = [None] * H
        # self.local_ARI = [None] * H
        self.local_FDR = [None] * H
        self.local_TPR = [None] * H
        self.local_Cm = [None] * H

        self.local_y_preds = [None] * H

        if self.real_label is not None:
            self.local_real_K = [None] * H
            self.local_real_Knum_stage1 = 0

        self.sub_oracle_labels = []

        self.local_runtime_st1 = [None] * H

        for h in range(H):
            a = self.machine_list[h]
            self.local_runtime_st1[h] = a.run_time

            self.local_RI[h] = a.RI
            self.local_FDR[h] = a.FDR
            self.local_TPR[h] = a.TPR
            self.local_Cm[h] = a.Cm1

            self.local_y_preds[h] = a.y_preds

            self.indicate_matrix[h] = a.Dh
            self.grouph[h] = a.grouph
            self.local_samplelabel[h] = a.samplelabel
            self.kforeachgroup_stage1.extend(a.kforeachgroup_stage1)
            self.local_kforeachgroup_stage1 = copy(a.kforeachgroup_stage1)

            if a.real_label is not None:
                self.local_real_K[h] = np.unique(a.real_label).shape[0]
                self.sub_oracle_labels.extend(a.real_label +
                                              self.local_real_Knum_stage1)
                self.local_real_Knum_stage1 = self.local_real_Knum_stage1 + self.local_real_K[
                    h]


            self.samplelabel_stage1.extend(self.local_samplelabel[h] +
                                           self.Knum_stage1)
            self.Knum_stage1 = self.Knum_stage1 + a.knum

            self.ini_local_Alpha[h] = (a.Alpha)
            self.ini_local_Knum[h] = (a.knum)
            self.ini_estbeta[h] = a.fiestbeta
            self.local_ini_EstTheta[h] = a.Esttheta

            if self.ini_local_Knum[h] > 1:
                self.AmTAm[h] = wm.createATA(
                    np.zeros((int(self.ini_local_Knum[h]), self.qq)))
                self.Am[h] = wm.createA(
                    np.zeros((int(self.ini_local_Knum[h]), self.qq)))[1]
                self.AmT[h] = self.Am[h].transpose()

                self.ini_local_vec_Alpha[h] = np.ravel(self.ini_local_Alpha[h])
                self.ini_vec_sigma[h] = self.Am[h].dot(
                    self.ini_local_vec_Alpha[h])
                self.ini_sigma[h] = self.ini_vec_sigma[h].reshape(
                    (int(len(self.ini_vec_sigma[h]) / self.qq), self.qq))
            else:
                self.Am[h] = 0
                self.AmTAm[h] = 0
                self.AmT[h] = 0

                self.ini_local_vec_Alpha[h] = np.ravel(self.ini_local_Alpha[h])

                self.ini_vec_sigma[h] = np.zeros(
                    self.ini_local_vec_Alpha[h].shape)
                self.ini_sigma[h] = self.ini_vec_sigma[h]

        self.runtime_st1 = np.max(self.local_runtime_st1)

        self.ini_barbeta = np.mean(self.ini_estbeta, axis=0)
        self.ini_EstTheta = np.concatenate(self.local_ini_EstTheta, axis=0)

        self.y_preds_st1 = np.concatenate(self.local_y_preds, axis=0)
        self.rmse_st1 = np.sqrt(mean_squared_error(self.oriY,
                                                   self.y_preds_st1))
        self.mae_st1 = mean_absolute_error(self.oriY, self.y_preds_st1)
        self.r_squared_manual_st1 = r2_score(self.oriY, self.y_preds_st1)

        if self.real_label is not None:
            self.RI_stage1 = wm.rand_score(self.real_label,
                                           self.samplelabel_stage1)
            self.ARI_stage1 = wm.adjusted_rand_score(self.real_label,
                                                     self.samplelabel_stage1)
            self.Cm_stage1 = wm.pair_confusion_matrix(self.real_label,
                                                      self.samplelabel_stage1)
            self.FDR_stage1 = self.Cm_stage1[0, 1] / (self.Cm_stage1[0, 1] +
                                                      self.Cm_stage1[1, 1])
            self.TPR_stage1 = self.Cm_stage1[1, 1] / (self.Cm_stage1[1, 0] +
                                                      self.Cm_stage1[1, 1])
        else:
            self.RI_stage1 = None
            self.ARI_stage1 = None
            self.FDR_stage1 = None
            self.TPR_stage1 = None
            self.Cm_stage1 = None

        # self.Knum_stage1 = sum(self.ini_local_Knum)

        self.AAstar, self.Smself, self.SmSm = wm.creatSM(
            H, self.qq, self.grouph, self.ini_local_Knum)
        ax = -1 / 10  
        self.Smself = [x**(ax) for x in self.Smself]
        self.SmSm = (self.SmSm)**(ax)

        self.xi = np.max(
            np.concatenate(
                ([y for x in self.Smself for y in x], self.SmSm))) + 1
        self.AAstarT = self.AAstar.transpose()

        self.ini_vvarpi = [0] * H
        for h in range(H):
            self.ini_vvarpi[h] = copy(self.ini_local_vec_Alpha[h])
        self.ini_vVarpi = np.array([y for x in self.ini_vvarpi for y in x])

        self.ini_vKappa = self.AAstar.dot(self.ini_vVarpi)
        self.Kappa = self.ini_vKappa.reshape(
            (int(len(self.ini_vKappa) / self.qq), self.qq))

        self.ZD = [0] * H
        self.XTXplusI_1 = [0] * H
        self.XXTXplusI_1 = [0] * H
        self.ZTQ = [0] * H
        self.MM1 = [0] * H
        self.ZTQY = [0] * H
        self.MM2 = [0] * H
        for h in range(H):
            a = self.machine_list[h]
            self.ZD[h] = wm.block_diag_matrix(a.oriZ).dot(
                np.kron(a.Dh, np.identity(self.qq)))

            if self.oriX is not None:
                self.XTXplusI_1[h] = np.linalg.inv(
                    np.dot(a.oriX.T, a.oriX) + self.xi * np.identity(self.pp))
                self.XXTXplusI_1[h] = np.dot(a.oriX, self.XTXplusI_1[h])
                Q_X = (np.identity(int(self.each_sample_size[h])) -
                       np.dot(self.XXTXplusI_1[h], a.oriX.T))
            else:
                Q_X = np.identity(int(self.each_sample_size[h]))

            self.ZTQ[h] = np.dot(self.ZD[h].T, Q_X)
            self.MM1[h] = np.linalg.inv(
                np.dot(self.ZTQ[h], self.ZD[h]) + self.xi * self.AmTAm[h] +
                self.xi * np.identity(int(self.ini_local_Knum[h]) * self.qq))
            self.ZTQY[h] = np.dot(self.ZTQ[h], a.oriY)
            if self.oriX is not None:
                self.MM2[h] = np.dot(self.ZD[h].T, self.XXTXplusI_1[h])

        self.MM3 = np.linalg.inv(
            self.AAstarT.dot(self.AAstar) +
            np.identity(self.Knum_stage1 * self.qq))

        if self.real_label is not None:
            self.sub_oracle_beta, self.sub_oracle_Zeta, self.sub_oracle_Theta = wm.oracle(
                self.oriZ, self.oriX, self.oriY, self.sub_oracle_labels)
            self.sub_oracle_RI = sa.rand_score(self.real_label,
                                               self.sub_oracle_labels)

            quasi_oracle_res = wm.quasi_oracle(self.oriZ, self.oriX, self.oriY,
                                               self.real_label,
                                               self.samplelabel_stage1)
            self.quasi_oracle_beta, self.quasi_oracle_Zeta, self.quasi_oracle_Theta = quasi_oracle_res[
                0]
            self.quasi_RI, self.quasi_FDR, self.quasi_TRP, self.quasi_Cm = quasi_oracle_res[
                1]

            self.oracle_beta, self.oracle_Zeta, self.oracle_Theta = wm.oracle(
                self.oriZ, self.oriX, self.oriY, self.real_label)
        else:
            self.sub_oracle_RI = None
            self.quasi_RI = None

        if self.real_beta is not None:
            self.mse_beta_st1 = np.mean((self.real_beta - self.ini_barbeta)**2)
            self.sub_oracle_mse_beta = np.mean(
                (self.real_beta - self.sub_oracle_beta)**2)
            self.quasi_oracle_mse_beta = np.mean(
                (self.real_beta - self.quasi_oracle_beta)**2)
            self.oracle_mse_beta = np.mean(
                (self.real_beta - self.oracle_beta)**2)
        else:
            self.mse_beta_st1 = None
            self.sub_oracle_mse_beta = None
            self.quasi_oracle_mse_beta = None
            self.oracle_mse_beta = None

        if self.real_theta is not None:
            self.mse_theta_st1 = np.mean(
                (self.real_theta.flatten() - self.ini_EstTheta.flatten())**2)
            self.sub_oracle_mse_theta = np.mean(
                (self.sub_oracle_Theta.flatten() -
                 self.real_theta.flatten())**2)
            self.quasi_oracle_mse_theta = np.mean(
                (self.quasi_oracle_Theta.flatten() -
                 self.real_theta.flatten())**2)
            self.oracle_mse_theta = np.mean(
                (self.oracle_Theta.flatten() - self.real_theta.flatten())**2)
        else:
            self.mse_theta_st1 = None
            self.sub_oracle_mse_theta = None
            self.quasi_oracle_mse_theta = None
            self.oracle_mse_theta = None


    def _str_stage1(self):
        self.name = f"Stage1 finished, {self.machine_num} local machines"

        if self.real_label is not None:
            self.real_K = np.unique(self.real_label).shape[0]
        else:
            self.real_K = None

        description = "Model:        %s\t\n" % self.name
        if self.train_switch_st1:

            description += "Size:        %s\n" % f"(N,q,p)=({self.N,self.qq,self.pp})"
            if self.real_label is not None:
                description += "Real.K&Est.K:  %s\n" % f"({self.real_K},{self.Knum_stage1})"
            else:
                description += "Est.K:     %s\n" % f"{self.Knum_stage1}"
            description += "Est.EachNum:  %s\n" % f"({np.array(self.kforeachgroup_stage1).astype(int)})"
            description += "RI_st1 & sub_RI & quasi_RI:            %s\n" % f"{self.RI_stage1, self.sub_oracle_RI, self.quasi_RI}"
            description += "FDR_st1:            %s\n" % f"{self.FDR_stage1}"
            description += "ARI_st1:            %s\n" % f"{self.ARI_stage1}"
            if self.real_beta is not None:
                description += "RMSE_beta_st1:    %s\n" % np.sqrt(
                    self.mse_beta_st1)
                description += "Sub_oracle_RMSE_beta: %s\n" % np.sqrt(
                    self.sub_oracle_mse_beta)
                description += "Quasi_oracle_RMSE_beta: %s\n" % np.sqrt(
                    self.quasi_oracle_mse_beta)
                description += "Oracle_RMSE_beta: %s\n" % np.sqrt(
                    self.oracle_mse_beta)
            if self.real_theta is not None:
                description += "RMSE_theta_st1:   %s\n" % np.sqrt(
                    self.mse_theta_st1)
                description += "Sub_oracle_RMSE_theta_st1: %s\n" % np.sqrt(
                    self.sub_oracle_mse_theta)
                description += "Quasi_oracle_RMSE_theta_st1: %s\n" % np.sqrt(
                    self.quasi_oracle_mse_theta)
                description += "Oracle_RMSE_theta_st1: %s\n" % np.sqrt(
                    self.oracle_mse_theta)
            # description += "BIC_st1:             %s\n" % self.bic_st1
            description += "RMSE_st1 & MAE_st1 & R2_st1:            %s\n" % f"{self.rmse_st1, self.mae_st1, self.r_squared_manual_st1}"
            description += "Run_time_st1 & Real_run_time:        %s\n" % f"{self.runtime_st1,sum(self.local_runtime_st1)}"

        else:
            description += "Not trained yet"
        return description

    def _analysis(self, Lam2, ini_estbeta=None, ini_local_Alpha=None):
        H = self.machine_num
        local_Lam2 = [Lam2] * H

        if ini_estbeta is None:
            estbeta = copy(self.ini_estbeta)
        else:
            estbeta = ini_estbeta
        barbeta = np.mean(estbeta, axis=0)

        if ini_local_Alpha is None:
            localAlpha = copy(self.ini_local_Alpha)
            sigma = copy(self.ini_sigma)
            vlocalAlpha = [None] * H
            vsigma = [None] * H
            for h in range(H):
                vlocalAlpha[h] = np.ravel(localAlpha[h])
                vsigma[h] = np.ravel(sigma[h])
            vvarpi = copy(self.ini_vvarpi)
            vVarpi = copy(self.ini_vVarpi)
            vKappa = copy(self.ini_vKappa)
            Kappa = copy(self.Kappa)

        else:
            localAlpha = ini_local_Alpha
            vlocalAlpha = [None] * H
            for h in range(H):
                vlocalAlpha[h] = np.ravel(localAlpha[h])
            vvarpi = copy(vlocalAlpha)
            vVarpi = np.array([y for x in vvarpi for y in x])
            vKappa = self.AAstar.dot(vVarpi)
            Kappa = vKappa.reshape((int(len(vKappa) / self.qq), self.qq))
            sigma = [None] * H
            vsigma = [None] * H
            for h in range(H):
                if self.ini_local_Knum[h] > 1:
                    vsigma[h] = self.Am[h].dot(vlocalAlpha[h])
                    sigma[h].reshape((int(len(vsigma[h]) / self.qq), self.qq))
                else:
                    vsigma[h] = np.zeros(vlocalAlpha[h].shape)
                    vsigma[h] = sigma[h]

        uu = [0] * H
        dd = [0] * H
        ll = [0] * H
        ss = np.zeros(vKappa.shape)
        frakd = copy(self.ini_sigma)
        vfrakd = copy(self.ini_vec_sigma)

        localized_gap = np.ones(H) * 100
        overall_gap = 100
        gap = 100
        counter = 0

        starttime = time()

        while gap >= self.tolerance or counter < self.T0:
            for h in range(H):
                a = self.machine_list[h]
                if self.oriX is not None:
                    uu[h] = uu[h] + self.xi * (estbeta[h] - barbeta)

                dd[h] = dd[h] + self.xi * (vlocalAlpha[h] - vvarpi[h])

                if self.ini_local_Knum[h] > 1:
                    if self.oriX is not None:
                        AA2 = self.ZTQY[h] + (self.xi * vvarpi[h] - dd[h]) - (
                            np.dot(self.MM2[h], self.xi * barbeta - uu[h])
                        ) + self.AmT[h].dot(self.xi * vsigma[h] - ll[h])
                    else:
                        AA2 = self.ZTQY[h] + (self.xi * vvarpi[h] -
                                              dd[h]) + self.AmT[h].dot(
                                                  self.xi * vsigma[h] - ll[h])
                elif self.ini_local_Knum[h] == 1:
                    if self.oriX is not None:
                        AA2 = self.ZTQY[h] + (self.xi * vvarpi[h] - dd[h]) - (
                            np.dot(self.MM2[h], self.xi * barbeta - uu[h]))
                    else:
                        AA2 = self.ZTQY[h] + (self.xi * vvarpi[h] - dd[h])
                vlocalAlpha[h] = np.dot(self.MM1[h], AA2)
                localAlpha[h] = vlocalAlpha[h].reshape(
                    (int(len(vlocalAlpha[h]) / self.qq), self.qq))

                if self.oriX is not None:
                    estbeta[h] = np.dot(
                        self.XTXplusI_1[h],
                        np.dot(a.oriX.T,
                               a.oriY - np.dot(self.ZD[h], vlocalAlpha[h])) +
                        self.xi * barbeta - uu[h])

                if self.ini_local_Knum[h] > 1:
                    mid_Delta_alpha = self.Am[h].dot(vlocalAlpha[h])

                    vfrakd[h] = mid_Delta_alpha + (1 / self.xi) * ll[h]
                    frakd[h] = vfrakd[h].reshape(
                        (int(len(vfrakd[h]) / self.qq), self.qq))
                    # MCP penalty

                    sigma[h] = wm.MCP_weighted(frakd[h], self.gamma,
                                               local_Lam2[h], self.xi,
                                               self.Smself[h])
                    vsigma[h] = sigma[h].ravel()

                    ll[h] = ll[h] + self.xi * (mid_Delta_alpha - vsigma[h])

                    oriresidual = np.array(mid_Delta_alpha - vsigma[h])

                    localized_gap[h] = np.linalg.norm(oriresidual.reshape(
                        int(len(oriresidual) / self.qq), self.qq),
                                                      ord=2)
                elif self.ini_local_Knum[h] == 1:
                    localized_gap[h] = 0
            if self.oriX is not None:
                barbeta = np.mean(estbeta, axis=0)

            combofdd = np.array([y for x in dd for y in x])
            vAlpha = np.array([y for x in vlocalAlpha for y in x])

            AA3 = vAlpha + (1 / self.xi) * combofdd + self.AAstarT.dot(
                vKappa - (1 / self.xi) * ss)
            vVarpi = np.array(self.MM3.dot(AA3))[0]
            Varpi = vVarpi.reshape(int(len(vVarpi) / self.qq), self.qq)
            varpi = np.split(Varpi, np.cumsum(self.ini_local_Knum)[:-1])
            for h in range(H):
                vvarpi[h] = varpi[h].ravel()

            mid_Delta_varpi = self.AAstar.dot(vVarpi)
            vfrakb = mid_Delta_varpi + (1 / self.xi) * ss
            frakb = vfrakb.reshape(int(len(vfrakb) / self.qq), self.qq)

            Kappa = wm.MCP_weighted(frakb, self.gamma, Lam2, self.xi,
                                    self.SmSm)
            vKappa = Kappa.ravel()
            ss = ss + self.xi * (mid_Delta_varpi - vKappa)

            oriresidual2 = mid_Delta_varpi - vKappa
            overall_gap = np.linalg.norm(oriresidual2.reshape(
                int(len(oriresidual2) / self.qq), self.qq),
                                         ord=2)

            gap = overall_gap + sum(localized_gap)

            counter = counter + 1
            if counter >= self.T1:
                break

        endtime = time()
        runtime_st2 = endtime - starttime

        fimatrix = np.ones((self.Knum_stage1, self.Knum_stage1)) * 100
        for i in range(self.Knum_stage1):
            for j in range(i, self.Knum_stage1):
                fimatrix[i, j] = np.linalg.norm(Varpi[i] - Varpi[j], 2)

        gresult = wm.Group(fimatrix, self.doorsill)

        Grou = []
        nnumm = np.insert(np.cumsum(self.each_sample_size), 0, 0)
        for h in range(H):
            for i in range(len(self.grouph[h])):
                Grou.append(
                    np.array(list(self.grouph[h][i])).astype(int) + nnumm[h])

        # 最终聚类结果，根据α的结果将对应的stage1的结果划为一类
        kforeachgroup_stage2 = np.zeros(len(gresult))
        samplegroups_stage2 = []
        # 给对应样本打上label
        samplelabel_stage2 = np.zeros(self.all_sample_size)
        for i in range(len(gresult)):
            zj = np.array([])
            for j in gresult[i]:
                zj = np.append(zj, Grou[j]).astype(int)
            samplegroups_stage2.append(zj)
            samplelabel_stage2[samplegroups_stage2[i]] = i
            kforeachgroup_stage2[i] = len(samplegroups_stage2[i])

        fiKnum = np.unique(samplelabel_stage2).shape[0]
        # ================================
        if self.get_res2 == 0:
            # 第一种方案根据估计的分组直接得到
            Z = wm.toZ(fimatrix, self.Knum_stage1, self.doorsill)
            invZ = np.linalg.inv(np.dot(Z.T, Z))
            invZZT = np.dot(invZ, Z.T)
            fiZeta = np.dot(invZZT, Varpi)
            Esttheta = np.ones((self.N, self.qq))
            for i in range(self.N):
                Esttheta[i] = fiZeta[int(samplelabel_stage2[i])]

        elif self.get_res2 == 1:
            # 第二种方案根据估计的分组逐组回归
            Esttheta, fiZeta, barbeta = wm.least_squares_fit_group_return(
                self.oriZ, self.oriX, self.oriY, samplelabel_stage2)

        elif self.get_res2 == 2:
            # 第三种方案根据估计的分组取oracle
            if fiKnum > 1:
                barbeta, fiZeta, Esttheta = wm.oracle(self.oriZ, self.oriX,
                                                      self.oriY,
                                                      samplelabel_stage2)
            else:
                fiZeta = np.mean(Esttheta, axis=0)
        # ================================
        resx = np.ones(self.N)
        for i in range(self.N):
            resx[i] = np.dot(self.oriZ[i], Esttheta[i])
        y_preds = np.zeros(self.N)
        if self.oriX is not None:
            y_preds = np.dot(self.oriX, barbeta) + resx
            Residual_stage2 = self.oriY - y_preds
        else:
            y_preds = resx
            Residual_stage2 = self.oriY - y_preds
        # print('Residual_stage2', Residual_stage2.shape)
        rmse = np.sqrt(mean_squared_error(self.oriY, y_preds))

        bic_stage2 = (
            np.log(np.dot(Residual_stage2.T, Residual_stage2) / self.N) +
            self.c2 * (np.log(self.N * self.qq + self.pp)) *
            (np.log(self.N) / self.N) * (fiKnum * self.qq + self.pp))
        # print('bic_stage2', np.dot(Residual_stage2.T, Residual_stage2).shape)

        if self.real_label is not None:
            # 计算ARI
            ARI = wm.adjusted_rand_score(self.real_label, samplelabel_stage2)
            # 计算RI
            RI = wm.rand_score(self.real_label, samplelabel_stage2)
            # 计算混淆矩阵
            Cm = wm.pair_confusion_matrix(self.real_label, samplelabel_stage2)
            FDR_stage2 = Cm[0, 1] / (Cm[0, 1] + Cm[1, 1])
            TPR = Cm[1, 1] / (Cm[1, 0] + Cm[1, 1])

        else:
            ARI = None
            RI = None
            Cm = None
            FDR_stage2 = None
            TPR = None

        return RI, ARI, FDR_stage2, Cm, TPR, runtime_st2, counter, gap, barbeta, fiZeta, Esttheta, localAlpha, samplelabel_stage2, samplegroups_stage2, kforeachgroup_stage2, fiKnum, bic_stage2, rmse, y_preds, Residual_stage2, Lam2

    def _calculate_summary_statiscs(self):
        fiKnum = self.fiKnum
        residuals = self.oriY - self.y_preds
        rmse = np.sqrt(mean_squared_error(self.oriY, self.y_preds))
        mae = mean_absolute_error(self.oriY, self.y_preds)
        r_squared_manual = r2_score(self.oriY, self.y_preds)
        X = self.oriX
        diagZ = wm.block_diag_matrix(self.oriZ)
        degree_of_freedom_Z = self.N - self.qq * self.fiKnum - self.pp
        sigma_hat_squared_Z = (residuals @ residuals) / degree_of_freedom_Z

        indicator_matrix = np.kron(
            np.eye(self.fiKnum)[np.array(self.samplelabel_stage2).astype(int)],
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
                self.fiZeta.shape)
        # print('s.e.', standard_errors_Z)

        t_value_Z = self.fiZeta / standard_errors_Z
        # print('t_value', t_value_Z)

        p_value_Z = 2 * (1 -
                         stats.t.cdf(np.abs(t_value_Z), degree_of_freedom_Z))
        # print('p_value', p_value_Z)

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
            # print('s.e.', standard_errors_X)

            t_value_X = self.barbeta / standard_errors_X
            # print('t_value', t_value_X)

            p_value_X = 2 * (
                1 - stats.t.cdf(np.abs(t_value_X), degree_of_freedom_X))
            # print('p_value', p_value_X)
        else:
            standard_errors_X = None
            t_value_X = None
            p_value_X = None

        return r_squared_manual, standard_errors_Z, t_value_Z, p_value_Z, standard_errors_X, t_value_X, p_value_X


    def _single_stage2(self, lam_list_st2=None, parallel=False, doorsill=None):
        if lam_list_st2 is None:
            lam_list_st2 = self.lam_list_st2

        if parallel:
            max_workers = round(multiprocessing.cpu_count() * 0.6)
            with multiprocessing.Pool(processes=max_workers) as pool:
                results = pool.map(self._analysis, lam_list_st2)

        else:
            # 串行计算
            results = [None] * len(lam_list_st2)
            for i in range(len(lam_list_st2)):
                results[i] = self._analysis(lam_list_st2[i])

        for i in range(len(lam_list_st2)):
            results[i] = self._analysis(lam_list_st2[i])

        Knum_list = np.zeros(len(lam_list_st2))
        BIC_list = np.zeros(len(lam_list_st2))
        RI_list = np.zeros(len(lam_list_st2))
        RMSE_list = np.zeros(len(lam_list_st2))
        self.kforeachgroup_stage2_list = [0] * len(lam_list_st2)
        for i in range(len(lam_list_st2)):
            RI, ARI, FDR_stage2, Cm, TPR, runtime_st2, counter, gap, barbeta, fiZeta, Esttheta, localAlpha, samplelabel_stage2, samplegroups_stage2, kforeachgroup_stage2, fiKnum, bic_stage2, rmse, y_preds, Residual_stage2, Lam2, = (
                results[i])

            Knum_list[i] = fiKnum
            BIC_list[i] = bic_stage2
            RI_list[i] = RI
            RMSE_list[i] = rmse
            self.kforeachgroup_stage2_list[i] = kforeachgroup_stage2

        best_index = np.argmin(BIC_list)
        # 取出最优结果
        best_result = results[best_index]
        self.best_lambda = lam_list_st2[best_index]

        self.bic_matrix = np.vstack(
            (lam_list_st2, Knum_list, RMSE_list, BIC_list)).T

        self.RI, self.ARI, self.FDR_stage2, self.Cm, self.TPR, self.runtime_st2, self.counter, self.gap, self.barbeta, self.fiZeta, self.Esttheta, self.localAlpha, self.samplelabel_stage2, self.samplegroups_stage2, self.kforeachgroup_stage2, self.fiKnum, self.bic_stage2, self.rmse, self.y_preds, self.Residual_stage2, self.Lam2 = best_result

        self.final_runtime = self.runtime_st1 + self.runtime_st2

        if self.real_label is not None:
            self.oracle_beta, self.oracle_Zeta, self.oracle_Theta = wm.oracle(
                self.oriZ, self.oriX, self.oriY, self.real_label)
        else:
            self.oracle_beta = None
            self.oracle_Zeta = None
            self.oracle_Theta = None

        if self.real_beta is not None:
            self.mse_beta_st2 = np.mean((self.real_beta - self.barbeta)**2)
            self.oracle_mse_beta = np.mean(
                (self.real_beta - self.oracle_beta)**2)
        else:
            self.mse_beta_st2 = None
            self.oracle_mse_beta = None

        if self.real_theta is not None:
            self.mse_theta_st2 = np.mean(
                (self.Esttheta.flatten() - self.real_theta.flatten())**2)
            self.oracle_mse_theta = np.mean(
                (self.oracle_Theta.flatten() - self.real_theta.flatten())**2)
        else:
            self.mse_theta_st2 = None
            self.oracle_mse_theta = None

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

        self.summary_statistics = {
            'RI_stage2:': self.RI,
            'ARI_stage2': self.ARI,
            'FDR_stage2': self.FDR_stage2,
            'TPR_stage2': self.TPR,
            'runtime_stage2': self.runtime_st2,
            'r_squared_manual': self.r_squared_manual,
            'se_rounded': self.se_rounded,
            'pvalue_rounded': self.pvalue_rounded,
            'se_intercept_rounded': self.se_intercept_rounded,
            'pvalue_intercept_rounded': self.pvalue_intercept_rounded
        }

        self.all_train_switch = True
        del self.Am
        del self.AmT
        del self.AmTAm

    def __str__(self):
        self.iniway = self.machine_list[0].iniway
        self.inik = [x.inik for x in self.machine_list]
        self.name = f"SA(lam={self.best_lambda}, iniway={self.iniway,self.inik})"

        if self.real_label is not None:
            self.real_K = np.unique(self.real_label).shape[0]

        description = "Model:        %s\t\n" % self.name
        if self.all_train_switch:

            description += "Size:        %s\n" % f"(N,q,p)=({self.N,self.qq,self.pp})"
            if self.real_label is not None:
                description += "Real.K&Est.K:  %s\n" % f"({self.real_K},{self.fiKnum})"
            else:
                description += "Est.K:     %s\n" % f"{self.fiKnum}"
            description += "Est.EachNum:  %s\n" % f"({(self.kforeachgroup_stage2).astype(int)})"
            description += "RI_st1 & sub_RI & quasi_RI & RI_st2:            %s\n" % f"{self.RI_stage1, self.sub_oracle_RI, self.quasi_RI,self.RI}"
            description += "FDR_st1 & FDR_st2:            %s\n" % f"{self.FDR_stage1,self.FDR_stage2}"
            description += "TPR_st1 & TPR_st2:            %s\n" % f"{self.TPR_stage1,self.TPR}"
            if self.real_beta is not None:
                description += "RMSE_beta:    %s\n" % np.sqrt(
                    self.mse_beta_st2)
                description += "Oracle_RMSE_beta: %s\n" % np.sqrt(
                    self.oracle_mse_beta)
            if self.real_theta is not None:
                description += "RMSE_theta:   %s\n" % np.sqrt(
                    self.mse_theta_st2)
                description += "Oracle_RMSE_theta: %s\n" % np.sqrt(
                    self.oracle_mse_theta)
            description += "R_squared:   %s\n" % self.r_squared_manual
            description += "se:   %s\n" % self.se_rounded
            description += "pvalue:   %s\n" % self.pvalue_rounded
            description += "BIC:             %s\n" % self.bic_stage2
            description += "Run_time & Run_time_st2 & Run_time_st1 & Real_run_time:             %s\n" % f"{self.final_runtime,self.runtime_st2,self.runtime_st1,sum(self.local_runtime_st1)}"

        else:
            description += "Not trained yet"
        return description
