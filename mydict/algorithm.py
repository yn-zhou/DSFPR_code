import stage1 as st1
import stage2 as st2
import numpy as np
import multiprocessing
from copy import copy


def train_local_machine(obj):
    obj.localized_train(parallel=False)
    print(obj.Alpha)
    return obj


class DSFPR:

    def __init__(self,
                 sample_index,
                 oriZ,
                 oriY,
                 oriX=None,
                 Feture_name_Z=None,
                 machine_num=None,
                 local_AN=None,
                 local_ANT=None,
                 local_ATA=None,
                 ini_esttheta=None,
                 lam_list_st1=[0.1, 0.5, 1, 5],
                 lam_list_st2=[0.1, 0.5, 1, 5],
                 real_label=None,
                 real_beta=None,
                 real_theta=None,
                 vertheta=3,
                 gamma=1,
                 doorsill_st1=0.01,
                 doorsill_st2=0.01,
                 tolerance=0.001,
                 T0=50,
                 T1=100,
                 c1=0.5,
                 get_res=1,
                 iniway='kmeans',
                 inik=None,
                 hot_start=False,
                 intercept_hete=True,
                 parallel_stage1=False,
                 parallel_stage2=False):

        if machine_num is None:
            machine_num = len(sample_index)
        if len(sample_index) > oriY.shape[0]:
            raise ValueError(
                'sample_index length must be less than or equal to oriY length'
            )

        if len(sample_index) != machine_num:
            raise ValueError(
                'sample_index length must be equal to machine_num')

        self.machine_num = machine_num
        self.real_X = copy(oriX)
        self.real_Z = copy(oriZ)
        self.real_beta = real_beta
        self.real_label = real_label
        self.real_theta = real_theta

        # 根据sample_index对样本进行划分
        local_oriZ = [None] * machine_num
        local_oriY = [None] * machine_num
        local_oriX = [None] * machine_num
        local_real_label = [None] * machine_num
        local_real_theta = [None] * machine_num
        for i in range(machine_num):
            local_oriY[i] = oriY[sample_index[i]]
            if oriX is not None:
                local_oriX[i] = oriX[sample_index[i]]
            if oriZ is not None:
                local_oriZ[i] = oriZ[sample_index[i]]
            if real_label is not None:
                local_real_label[i] = real_label[sample_index[i]]
            if real_theta is not None:
                local_real_theta[i] = real_theta[sample_index[i]]

        self.local_oriZ = local_oriZ
        self.local_oriY = local_oriY
        self.local_oriX = local_oriX
        self.local_real_label = local_real_label
        self.local_real_theta = local_real_theta

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

        self.oriX = copy(oriX)
        self.oriZ = copy(oriZ)
        self.oriY = copy(oriY.ravel())

        self.Feture_name_Z = Feture_name_Z

        if local_AN is None:
            local_AN = [None] * machine_num
        if local_ANT is None:
            local_ANT = [None] * machine_num
        if local_ATA is None:
            local_ATA = [None] * machine_num
        if ini_esttheta is None:
            ini_esttheta = [None] * machine_num
        self.local_AN = local_AN
        self.local_ANT = local_ANT
        self.local_ATA = local_ATA
        self.ini_esttheta = ini_esttheta

        self.vertheta = vertheta
        self.gamma = gamma
        self.doorsill_st1 = doorsill_st1
        self.doorsill_st2 = doorsill_st2
        self.tolerance = tolerance
        self.T0 = T0
        self.T1 = T1
        self.c1 = c1
        self.get_res = get_res
        self.iniway = iniway
        self.inik = inik
        self.hot_start = hot_start
        self.intercept_hete = intercept_hete
        self.parallel_stage1 = parallel_stage1
        self.parallel_stage2 = parallel_stage2
        self.lam_list_st1 = lam_list_st1
        self.lam_list_st2 = lam_list_st2

        self.pre_stage1_sitch = False
        self.stage1_sitch = False
        self.pre_stage2_sitch = False
        self.stage2_sitch = False

    def pre_stage1(self):
        # 创建stage1的各本地机器，及其初始化
        self.machine_list = [None] * self.machine_num
        for h in range(self.machine_num):
            self.machine_list[h] = st1.local_machine(
                machine_index=h,
                oriZ=self.local_oriZ[h],
                oriY=self.local_oriY[h],
                oriX=self.local_oriX[h],
                Feture_name_Z=self.Feture_name_Z,
                AN=self.local_AN[h],
                ANT=self.local_ANT[h],
                ATA=self.local_ATA[h],
                ini_esttheta=self.ini_esttheta[h],
                lam_list=self.lam_list_st1,
                real_label=self.local_real_label[h],
                real_beta=self.real_beta,
                real_theta=self.local_real_theta[h],
                vertheta=self.vertheta,
                gamma=self.gamma,
                doorsill=self.doorsill_st1,
                tolerance=self.tolerance,
                T0=self.T0,
                T1=self.T1,
                c1=self.c1,
                get_res=self.get_res,
                iniway=self.iniway,
                inik=self.inik,
                hot_start=self.hot_start,
                intercept_hete=self.intercept_hete)

        self.pre_stage1_sitch = True



    def stage1(self, lam_list_st1=None, parallel_stage1=None, keep_st1=False):
        if lam_list_st1 is None:
            lam_list_st1 = self.lam_list_st1

        if self.pre_stage1_sitch is False:
            self.pre_stage1()
            print('pre_stage1 is not executed, executing pre_stage1 first')

        if self.stage1_sitch is True and keep_st1 is False:
            print('stage1 is already executed, no need to execute again')
        else:
            if parallel_stage1 is None:
                parallel_stage1 = self.parallel_stage1
            if parallel_stage1 is True:

                max_workers = round(multiprocessing.cpu_count() * 0.6)
                with multiprocessing.Pool(max_workers) as pool:
                    updated_obj_list = pool.map(train_local_machine,
                                                self.machine_list)

                self.machine_list = updated_obj_list
            else:
                for h in range(self.machine_num):
                    self.machine_list[h].localized_train(lam_list=lam_list_st1,
                                                         parallel=True)

            self.stage1_sitch = True

    def pre_stage2(self, lam_list_st2=None):
        self.master_machine = st2.master_machine(
            machine_list=self.machine_list,
            oriZ=self.oriZ,
            oriY=self.oriY,
            oriX=self.oriX,
            lam_list_st2=lam_list_st2,
            real_label=self.real_label,
            real_beta=self.real_beta,
            real_theta=self.real_theta,
            vertheta=self.vertheta,
            gamma=self.gamma,
            doorsill=self.doorsill_st2,
            tolerance=self.tolerance,
            T0=self.T0,
            T1=self.T1,
            c1=self.c1,
            get_res=self.get_res,
            hot_start=self.hot_start)

        self.pre_stage2_sitch = True
        self.master_machine.train_switch_st1 = True
        print('pre_stage2 is executed, please check the results')

        print(self.master_machine._str_stage1())
        print('~~~~~~~~~~~~~~stage boundary line~~~~~~~~~~~~~~~~')

    def stage2(self, lam_list_st2=None, parallel_stage2=None, keep_st2=False):
        if self.pre_stage2_sitch is False:
            self.pre_stage2(lam_list_st2)
            print('pre_stage2 is not executed, executing pre_stage2 first')
        else:
            print(self.master_machine._str_stage1())
            print('~~~~~~~~~~~~~~stage boundary line~~~~~~~~~~~~~~~~')

        if self.stage2_sitch is True and keep_st2 is False:
            print('stage2 is already executed, no need to execute again')
        else:
            if parallel_stage2 is None:
                parallel_stage2 = self.parallel_stage2
            if lam_list_st2 is None:
                lam_list_st2 = self.lam_list_st2
            self.master_machine._single_stage2(lam_list_st2=lam_list_st2,
                                               parallel=parallel_stage2)
            print(self.master_machine)
            self.stage2_sitch = True
            print('stage2 is executed, please check the results')

    def fit(self,
            lam_list_st1=None,
            lam_list_st2=None,
            parallel_st1=None,
            parallel_stage2=None):
        self.stage1(lam_list_st1=lam_list_st1, parallel_stage1=parallel_st1)
        self.stage2(lam_list_st2=lam_list_st2, parallel_stage2=parallel_stage2)
        print('=========simulated boundary line===========')
        print('fit is executed, please check the results')


class DSFPR_DF(DSFPR):


    def __init__(self,
                 dataframe,
                 splite_index,
                 Feture_name_Z,
                 Feature_name_Y,
                 Feture_name_X=None,
                 machine_num=None,
                 local_AN=None,
                 local_ANT=None,
                 local_ATA=None,
                 ini_esttheta=None,
                 lam_list_st1=[0.1, 0.5, 1, 5],
                 lam_list_st2=[0.1, 0.5, 1, 5],
                 real_label=None,
                 real_beta=None,
                 real_theta=None,
                 vertheta=3,
                 gamma=1,
                 doorsill_st1=0.01,
                 doorsill_st2=0.01,
                 tolerance=0.001,
                 T0=50,
                 T1=100,
                 c1=0.5,
                 get_res=1,
                 iniway='kmeans',
                 inik=None,
                 hot_start=False,
                 intercept_hete=True,
                 parallel_stage1=False,
                 parallel_stage2=False):

        oriZ = dataframe[Feture_name_Z].values
        oriY = dataframe[Feature_name_Y].values
        if Feture_name_X is not None:
            oriX = dataframe[Feture_name_X].values
        else:
            oriX = None

        super().__init__(sample_index=splite_index,
                         oriZ=oriZ,
                         oriY=oriY,
                         oriX=oriX,
                         Feture_name_Z=Feture_name_Z,
                         machine_num=machine_num,
                         local_AN=local_AN,
                         local_ANT=local_ANT,
                         local_ATA=local_ATA,
                         ini_esttheta=ini_esttheta,
                         lam_list_st1=lam_list_st1,
                         lam_list_st2=lam_list_st2,
                         real_label=real_label,
                         real_beta=real_beta,
                         real_theta=real_theta,
                         vertheta=vertheta,
                         gamma=gamma,
                         doorsill_st1=doorsill_st1,
                         doorsill_st2=doorsill_st2,
                         tolerance=tolerance,
                         T0=T0,
                         T1=T1,
                         c1=c1,
                         get_res=get_res,
                         iniway=iniway,
                         inik=inik,
                         hot_start=hot_start,
                         intercept_hete=intercept_hete,
                         parallel_stage1=parallel_stage1,
                         parallel_stage2=parallel_stage2)
