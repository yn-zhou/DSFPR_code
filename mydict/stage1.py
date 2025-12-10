import numpy as np
import main_sa as sa



class local_machine(sa.Subgroups_analysis):

    def __init__(self,
                 machine_index,
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
                 c1=0.5,
                 get_res=1,
                 iniway='mixreg',
                 inik=None,
                 hot_start=False,
                 intercept_hete=True):

        self.machine_index = machine_index

        super().__init__(oriZ=oriZ,
                         oriY=oriY,
                         oriX=oriX,
                         Feture_name_Z=Feture_name_Z,
                         AN=AN,
                         ANT=ANT,
                         ATA=ATA,
                         ini_esttheta=ini_esttheta,
                         lam_list=lam_list,
                         real_label=real_label,
                         real_beta=real_beta,
                         real_theta=real_theta,
                         vertheta=vertheta,
                         gamma=gamma,
                         doorsill=doorsill,
                         tolerance=tolerance,
                         T0=T0,
                         T1=T1,
                         c1=c1,
                         get_res=get_res,
                         iniway=iniway,
                         inik=inik,
                         hot_start=hot_start,
                         intercept_hete=intercept_hete)

    def localized_train(self, lam_list=None, parallel=False):
        if lam_list is None:
            lam_list = self.lam_list

        self._single_sa(lam_list=lam_list, parallel=parallel)
        print('Machine_', self.machine_index, 'have trained successfully!')
        print(self)
        print('-------------单机分割线-----------------')
