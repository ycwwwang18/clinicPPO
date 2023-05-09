from copy import deepcopy


class Server:
    """项目服务台"""
    def __init__(self, idLabel, capacity, avg_service_time, acc_wait_time):
        """
        :param idLabel: 服务台所属的项目种类，包括检查项目A-G、问诊项目H1-H3（问诊项目的多少取决于医生的数量）
        :param capacity: 该项目并行服务台的数量
        :param avg_service_time: 该项目的平均服务时间
        """
        self.idLabel = idLabel
        self.capacity = capacity
        self.avg_service_time = avg_service_time
        self.acc_wait_time = acc_wait_time
        self.service_time = 0  # 该服务台的服务总时长
        self.service_end_time = 0  # 该服务台的预计服务结束时间
        self.status = 'idle'  # 服务台的状态
        self.que_len = 0  # 服务台的排队长

    def reset(self):
        # 重置服务台的状态
        self.service_time = 0
        self.service_end_time = 0
        self.status = 'idle'
        self.que_len = 0


class Patient:
    """患者"""
    def __init__(self, idLabel, doctor, service_completeness: dict, arrive_slot=0, acceptable_wait_time=0):
        """
        :param idLabel: 患者编号
        :param doctor: 患者的签约医生
        :param service_completeness: 服务完成情况，未完成为1
        :param arrive_slot: 患者的预约到达时间
        """
        self.idLabel = idLabel
        self.doctor = doctor
        self.service_completeness = service_completeness  # 患者的项目完成情况
        self.service_completeness_reset = deepcopy(service_completeness)  # 用于重置
        self.arrive_slot = arrive_slot
        self.wait_time = 0  # 患者在产检中的等待时间
        self.status = 'not_arrive'
        self.wait_server = 0
        self.service_time = 0  # 患者在系统当前的已接受过服务的服务时长
        self.service_start_time = 0  # 患者最近一次服务的开始时间
        self.arrive_time = -1  # 患者到达时间
        self.acceptable_wait_time = acceptable_wait_time  # 可接受的总等待时间
        self.acceptable_wait_time_reset = acceptable_wait_time  # 用于重置

    def servicePathDone(self):
        completeness = list(self.service_completeness.values())
        done = all(item <= 0 for item in completeness)  # 如果所有项目都是0，说明已经完成
        return done

    def getInitialWaitTime(self):
        wait_time = deepcopy(self.service_completeness)
        for key in wait_time.keys():
            wait_time[key] = 0
        return wait_time

    def reset(self):
        # 重置患者的状态
        self.wait_time = 0
        self.status = 'not_arrive'
        self.service_time = 0
        self.service_start_time = 0
        self.arrive_time = -1
        self.service_completeness = deepcopy(self.service_completeness_reset)
        self.acceptable_wait_time = self.acceptable_wait_time_reset
