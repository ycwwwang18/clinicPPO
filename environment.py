import random
import simpy
import numpy as np
import pandas as pd
from clinicElements import Server, Patient


class Environment:
    """产科门诊仿真环境"""

    def __init__(self, patient_path, server_path, avg_arrive_time, observe_flag=0):
        # 生成数据
        self.acceptable_wait_time = []
        self.avg_service_time = {}
        self.server_capacity = {}
        self.patients, self.servers = self.dataGen(patient_path, server_path)
        self.avg_arrive_time = avg_arrive_time
        self.n_patients = len(self.patients)
        self.n_servers = len(self.servers)
        self.requestServerEvent = False
        self.requestPatient = None
        self.server = None
        self.done = False
        self.observe_flag = observe_flag

        self.n_observe = 6

    @staticmethod
    def seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    def dataGen(self, patient_path, server_path):
        """将原始数据构造成实例化对象"""

        def gen_service_completeness(p):
            # 将path转换为0-1编码的形式
            server_names = list(server_data['ID Label'])
            values = [0 for _ in range(len(server_names))]
            service_completeness_dict = dict(zip(server_names, values))
            for item in p:
                service_completeness_dict[item] += 1
            return service_completeness_dict

        servers = []  # 传给Patient server_idLabel: server object
        patients = []  # 传给PatientGenerator [<Patient>,]
        patient_data = pd.read_excel(patient_path)
        server_data = pd.read_excel(server_path)
        self.acceptable_wait_time = list(server_data['Acceptable Wait Time(min)'])
        for _, row in server_data.iterrows():
            idLabel = row['ID Label']
            capacity = row['Capacity']
            avg_service_time = row['Average Service Time(min)']
            acc_wait_time = row['Acceptable Wait Time(min)']
            server = Server(idLabel, capacity, avg_service_time, acc_wait_time)
            servers.append(server)
            self.avg_service_time[idLabel] = avg_service_time
            self.server_capacity[idLabel] = capacity

        for _, row in patient_data.iterrows():
            idLabel = row['Patient ID']
            doctor = row['Doctor']
            path = row['Path'].split("-")
            service_completeness = gen_service_completeness(path)
            acceptable_wait_time = np.dot(np.array(self.acceptable_wait_time), np.array(list(service_completeness.values())))
            arrival_slot = 0 if row['Arrival Time Slot'] == np.NAN else row['Arrival Time Slot']
            patient = Patient(idLabel, doctor, service_completeness, arrival_slot, acceptable_wait_time)
            patients.append(patient)

        return patients, servers

    def reset(self):
        """initialize the environment(when the first patient arrives), return the current state"""
        self.env = simpy.Environment()
        self.requestServerEvent = False
        # 重置患者状态
        for patient in self.patients:
            patient.reset()
        # 生成服务台
        self.resources = {}
        for server in self.servers:
            self.resources[server.idLabel] = simpy.Resource(self.env, server.capacity)
            server.reset()
        self.env.process(self.mainProcess())
        self.env.step()
        return self.observe()

    def step(self, action, observation):
        """step action, return new observation and reward and done"""
        self.server = self.servers[action]
        while not self.requestServerEvent:
            try:
                self.env.step()
            except simpy.core.EmptySchedule:
                break
        self.requestServerEvent = False
        feature_vector_, observation_ = self.observe()
        reward = self.reward(observation, observation_)
        done = self.isDone()
        return feature_vector_, reward, done, observation_

    def mainProcess(self):
        """产检门诊主流程"""
        for i, patient in enumerate(self.patients):
            # 患者进行产检服务
            patient.arrive_time = self.env.now
            if i == 0:
                self.requestPatient = patient
            self.env.process(self.doServicePath(patient))
            # 患者到达事件
            random_arrival_duration = self.avg_arrive_time + np.random.uniform(-1, 1)
            # print(f'random arrival duration: {random_arrival_duration}')
            new_patient_arrive = self.env.timeout(delay=random_arrival_duration)
            # print(f"患者{patient.idLabel}在{self.env.now}到达产科门诊")
            yield new_patient_arrive

    def doServicePath(self, patient):
        """患者的整个产检流程"""
        count = 1
        while True:
            if patient.servicePathDone():
                break
            if (patient.idLabel == 1) & (count == 1):
                self.requestServerEvent = False
                self.requestPatient = patient
            else:
                self.requestServerEvent = True
                self.requestPatient = patient
            yield self.env.process(self.doService(self.env, patient))
            count += 1

    def doService(self, env, patient):
        """患者请求并进行一项服务"""
        # 第一个患者的第一项服务不停，后面都停
        request_server = self.server.idLabel
        patient.service_completeness[request_server] -= 1  # 标记该服务已经完成
        patient.status = 'waiting'
        patient.wait_server = request_server
        with self.resources[request_server].request() as rq:  # 请求服务
            yield rq  # 等待资源被释放
            patient.status = 'servicing'
            patient.service_start_time = env.now
            if self.server.avg_service_time < 10:
                random_service_duration = self.server.avg_service_time + np.random.uniform(-1,1)
            else:
                random_service_duration = self.server.avg_service_time + np.random.uniform(-5,5)
            # random_service_duration = self.server.avg_service_time
            # print(f'average service duration: {self.server.avg_service_time}, random service duration: {random_service_duration}')
            self.server.service_time += random_service_duration
            self.server.service_end_time = patient.service_start_time + random_service_duration
            yield env.timeout(random_service_duration)  # 服务台提供服务
            patient.service_time += random_service_duration
            patient.status = 'requesting'

    @staticmethod
    def reward(observation, observation_):
        """stage cost function"""
        RF_a, RF_e, U_avg = observation['RF_a'], observation['RF_e'], observation['U_avg']
        RF_a_, RF_e_, U_avg_ = observation_['RF_a'], observation_['RF_e'], observation_['U_avg']
        if RF_a_ < RF_a:
            return 1
        elif RF_a_ > RF_a:
            return -1
        elif RF_e_ < RF_e:
            return 1
        elif RF_e_ > RF_e:
            return -1
        elif U_avg_ > U_avg:
            return 1
        elif U_avg_ > U_avg * 0.95:
            return 0
        else:
            return -1

    def getRemainService(self):
        """get the remaining services of all patients"""
        completeness = []
        for patient in self.patients:
            server = []
            for key, value in patient.service_completeness.items():
                if value == 1:
                    server.append(key)
            completeness.append(server)
        return completeness

    def observeCompleteness(self):
        """return all patients' completeness rate"""
        completeness_rate_list = []
        for patient in self.patients:
            incomplete_num = sum(list(patient.service_completeness.values()))
            service_num = sum(list(patient.service_completeness_reset.values()))
            completeness_rate = 1 - incomplete_num / service_num
            completeness_rate_list.append(completeness_rate)
        return completeness_rate_list

    def getWaitTime(self):
        """return current wait time of all patients"""
        wait_time_list = []
        for patient in self.patients:
            if patient.status == 'waiting':
                patient.wait_time = self.env.now - patient.service_time - patient.arrive_time
            elif patient.status == 'servicing':
                patient.wait_time = patient.service_start_time - patient.service_time - patient.arrive_time
            wait_time_list.append(patient.wait_time)
        return wait_time_list

    def getQueueLen(self):
        """return the queue length of all servers"""
        queue_len = {}
        for server_id, resource in self.resources.items():
            queue_len[server_id] = len(resource.queue)
        return queue_len

    def getAvgServiceTime(self):
        """return the average service time of all servers"""
        return self.avg_service_time

    def getServerCapacity(self):
        """return the capacity of all servers"""
        return self.server_capacity

    def getRemainPatientNum(self,remain_service):
        """return the number of patients who still need service from the servers"""
        remain_service = np.array(remain_service, dtype=object)
        remain_service = list(remain_service.flatten())
        remain_patient_num = {}
        for server_id in self.resources.keys():
            remain_patient_num[server_id] = remain_service.count(server_id)
        return remain_patient_num

    def getEstWorkLoad(self, remain_service):
        """return the estimated work load of all servers"""
        est_work_load = {}
        remain_patient_num = self.getRemainPatientNum(remain_service)
        for server_idLabel in self.resources.keys():
            est_work_load[server_idLabel] = remain_patient_num[server_idLabel] * self.avg_service_time[server_idLabel] / self.server_capacity[server_idLabel]
        return est_work_load

    def getPatientWaitTimeEst(self, patient_id, cur_wait_time, remain_service, queue_len, est_work_load):
        """return the estimated waiting time of the given patient"""
        if not remain_service:
            return cur_wait_time
        elif self.patients[patient_id].status == 'waiting':
            server_id = self.patients[patient_id].wait_server
            cur_server_wait_time = queue_len[server_id] * self.avg_service_time[server_id] / self.server_capacity[server_id]
            future_wait_time = 0
            for server_idLabel in remain_service:
                future_wait_time += est_work_load[server_idLabel]
            return cur_wait_time + cur_server_wait_time + future_wait_time
        elif self.patients[patient_id].status == 'servicing' or self.patients[patient_id].status == 'requesting' or self.patients[patient_id].status == 'not_arrive':
            future_wait_time = 0
            for server_idLabel in remain_service:
                future_wait_time += est_work_load[server_idLabel]
            return cur_wait_time + future_wait_time

    def getEstimatedRF(self, cur_wait_time):
        """return RF_e(t)"""
        remain_service = self.getRemainService()
        queue_len = self.getQueueLen()
        est_work_load = self.getEstWorkLoad(remain_service)
        rf_num = 0
        for i in range(self.n_patients):
            total_est_wait_time = self.getPatientWaitTimeEst(i, cur_wait_time[i], remain_service[i], queue_len, est_work_load)
            if total_est_wait_time > self.patients[i].acceptable_wait_time:
                rf_num += 1
        return rf_num / self.n_patients

    def getRealRF(self, cur_wait_time):
        """return RF_a(t)"""
        rf_num = 0
        for i in range(self.n_patients):
            if cur_wait_time[i] > self.patients[i].acceptable_wait_time:
                rf_num += 1
        return rf_num / self.n_patients

    def observeServerUtilization(self):
        """return current utilization rate of each server"""
        util_rate_list = []
        for server in self.servers:
            if server.service_end_time != 0:
                if self.env.now < server.service_end_time:
                    util_rate = server.service_time / server.service_end_time
                else: util_rate = server.service_time / self.env.now
            else: util_rate = 0
            util_rate_list.append(util_rate)
        return util_rate_list

    def isDone(self):
        """return whether the episode is done or not"""
        completeness = pd.DataFrame()
        for patient in self.patients:
            completeness = completeness.append(patient.service_completeness, ignore_index=True)
        completeness = np.array(completeness)
        if np.any(completeness < 0):
            print(completeness)
        return np.all(completeness == 0)

    def observe(self):
        """map state into feature vector, return tensor"""
        feature_vector = []
        util_rate = self.observeServerUtilization()
        U_avg = np.mean(util_rate)
        U_std = np.std(util_rate)
        completion = self.observeCompleteness()
        C_avg = np.mean(completion)
        C_std = np.std(completion)
        wait_time = self.getWaitTime()
        RF_a = self.getRealRF(wait_time)
        RF_e = self.getEstimatedRF(wait_time)
        feature_vector.append(U_avg)
        feature_vector.append(U_std)
        feature_vector.append(C_avg)
        feature_vector.append(C_std)
        feature_vector.append(RF_a)
        feature_vector.append(RF_e)

        completion = list(self.patients[self.requestPatient.idLabel - 1].service_completeness.values())
        queue_len = []
        for i, resource in enumerate(self.resources.values()):
            queue_len.append(len(resource.queue) * self.servers[i].avg_service_time)
        remain_service = self.getRemainService()
        est_work_load = self.getEstWorkLoad(remain_service)
        observation = {'completion': completion,
                       'queue_length': queue_len,
                       'avg_service_time': list(self.avg_service_time.values()),
                       'RF_a': RF_a,
                       'RF_e': RF_e,
                       'U_avg': U_avg,
                       'work_load': est_work_load
                       }
        return feature_vector, observation


if __name__ == '__main__':
    PATIENT_PATH = './data/patient_21.xlsx'  # 患者信息
    SERVER_PATH = './data/server information.xlsx'  # 服务台信息
    AVG_ARRIVE_TIME = 1  # 患者平均到达间隔
    clinic_env = Environment(PATIENT_PATH, SERVER_PATH, AVG_ARRIVE_TIME, observe_flag=2)
    done = False
    obs = clinic_env.reset()
    while not done:
        print(f'请求患者为：{clinic_env.requestPatient.idLabel}')
        patient_done = clinic_env.requestPatient.servicePathDone()
        print('患者完成情况：', patient_done)
        if patient_done:
            print(clinic_env.requestPatient.idLabel)
            print(patient_done)
            break
        try:
            a = random.choice(
                [i for i, v in enumerate(clinic_env.requestPatient.service_completeness.values()) if v != 0])
        except IndexError:
            print(patient_done)
        print(clinic_env.observeCompleteness())
        print(clinic_env.observeServerUtilization())
