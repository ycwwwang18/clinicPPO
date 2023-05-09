class Cal:
    def __init__(self, now, patients, servers, resources, avg_service_time, server_capacity):
        self.now = now
        self.patients = patients
        self.servers = servers
        self.resources = resources
        self.avg_service_time = avg_service_time
        self.server_capacity = server_capacity
        self.n_patients = len(self.patients)

        self.remain_service = self.getRemainService()
        self.completion_rate = self.observeCompleteness()
        self.cur_wait_time = self.getWaitTime()
        self.total_waiting_time = sum(self.cur_wait_time)
        self.que_len = self.getQueueLen()
        self.que_time = self.getQueueTime()
        self.remain_patient_num = self.getRemainPatientNum()
        self.est_work_load = self.getEstWorkLoad()
        self.est_wait_time = self.getEstWaitTime()
        self.RF_e = self.getEstimatedRF()
        self.RF_a = self.getRealRF()
        self.util_rate_list = self.observeServerUtilization()

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
                patient.wait_time = self.now - patient.service_time - patient.arrive_time
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

    def getQueueTime(self):
        """return the queue time of all servers"""
        que_times = []
        i = 0
        for server_id, resource in self.resources.items():
            server = self.servers[i]
            if server.service_end_time < self.now:
                queue_time = 0
            elif server.service_end_time == self.now:
                if len(resource.queue) != 0:  # 改成queue len
                    queue_time = len(resource.queue) * self.avg_service_time[server_id] + self.avg_service_time[server_id]
                else:
                    queue_time = 0
            else:
                queue_time = len(resource.queue) * self.avg_service_time[server_id] + (server.service_end_time - self.now)

            que_times.append(queue_time)
            i += 1
        return que_times

    def getRemainPatientNum(self):
        """return the number of patients who still need service from the servers"""
        remain_service = [item for lis in self.remain_service for item in lis]  # 修复将二维list转为一维list的错误
        remain_patient_num = {}
        for server_id in self.resources.keys():
            remain_patient_num[server_id] = remain_service.count(server_id)
        return remain_patient_num

    def getEstWorkLoad(self):
        """return the estimated work load of all servers"""
        est_work_load = {}
        remain_patient_num = self.remain_patient_num
        for server_idLabel in self.resources.keys():
            est_work_load[server_idLabel] = remain_patient_num[server_idLabel] * self.avg_service_time[server_idLabel] / self.server_capacity[server_idLabel]
        return est_work_load

    def getPatientWaitTimeEst(self, patient_id):
        """return the estimated waiting time of the given patient"""
        remain_service = self.remain_service[patient_id]
        cur_wait_time = self.cur_wait_time[patient_id]
        if not remain_service:
            return cur_wait_time
        elif self.patients[patient_id].status == 'waiting':
            server_id = self.patients[patient_id].wait_server
            cur_server_wait_time = self.que_len[server_id] * self.avg_service_time[server_id] / self.server_capacity[server_id]
            future_wait_time = 0
            for server_idLabel in remain_service:
                future_wait_time += self.est_work_load[server_idLabel]
            return cur_wait_time + cur_server_wait_time + future_wait_time
        elif self.patients[patient_id].status == 'servicing' or self.patients[patient_id].status == 'requesting' or self.patients[patient_id].status == 'not_arrive':
            future_wait_time = 0
            for server_idLabel in remain_service:
                future_wait_time += self.est_work_load[server_idLabel] / 2  # / self.remain_patient_num[server_idLabel]
            return cur_wait_time + future_wait_time

    def getEstWaitTime(self):
        est_wait_time = []
        for i in range(self.n_patients):
            est_wait_time.append(self.getPatientWaitTimeEst(i))
        return est_wait_time

    def getEstimatedRF(self):
        """return RF_e(t)"""
        rf_num = 0
        for i in range(self.n_patients):
            total_est_wait_time = self.est_wait_time[i]
            if total_est_wait_time > self.patients[i].acceptable_wait_time:
                rf_num += 1
        return rf_num / self.n_patients

    def getRealRF(self):
        """return RF_a(t)"""
        rf_num = 0
        for i in range(self.n_patients):
            if self.cur_wait_time[i] > self.patients[i].acceptable_wait_time:
                rf_num += 1
        return rf_num / self.n_patients

    def observeServerUtilization(self):
        """return current utilization rate of each server"""
        util_rate_list = []
        for server in self.servers:
            if server.service_end_time != 0:
                if self.now < server.service_end_time:
                    util_rate = server.service_time / server.service_end_time
                else:
                    util_rate = server.service_time / self.now
            else:
                util_rate = 0
            util_rate_list.append(util_rate)
        return util_rate_list
