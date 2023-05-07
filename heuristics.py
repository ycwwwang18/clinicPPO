import numpy as np


class Heuristic(object):
    def __init__(self, heuristic_ind, observation):
        if heuristic_ind == 0:
            self.action = self.LMS(observation)
        elif heuristic_ind == 1:
            self.action = self.SMS(observation)
        elif heuristic_ind == 2:
            self.action = self.LEW(observation)
        elif heuristic_ind == 3:
            self.action = self.SEW(observation)
        elif heuristic_ind == 4:
            self.action = self.LEWMS(observation)
        elif heuristic_ind == 5:
            self.action = self.SEWMS(observation)
        elif heuristic_ind == 6:
            self.action = self.MW(observation)

    @staticmethod
    def LMS(observation):
        """最长平均服务时间优先"""
        completion = np.array(observation['completion'])
        avg_service_time = np.array(observation['avg_service_time'])
        eligible = completion * avg_service_time
        return np.argmax(eligible)

    @staticmethod
    def SMS(observation):
        """最短平均服务时间优先"""
        completion = np.array(observation['completion'])
        avg_service_time = np.array(observation['avg_service_time'])
        completion = np.array(list(map(lambda x: 99999 if x == 0 else x, completion)))
        eligible = completion * avg_service_time
        return np.argmin(eligible)

    @staticmethod
    def LEW(observation):
        """最长期望等待时间优先"""
        completion = np.array(observation['completion'])
        queue_len = np.array(observation['queue_length']) + 1
        eligible = completion * queue_len
        return np.argmax(eligible)

    @staticmethod
    def SEW(observation):
        """最短期望等待时间优先"""
        completion = np.array(observation['completion'])
        queue_len = np.array(observation['queue_length']) + 1  # +1是防止在全0的情况下，选择了已完成的项目
        completion = np.array(list(map(lambda x: 99999 if x == 0 else x, completion)))
        eligible = completion * queue_len
        return np.argmin(eligible)

    @staticmethod
    def LEWMS(observation):
        """最长期望等待时间+平均服务时间优先"""
        completion = np.array(observation['completion'])
        avg_service_time = np.array(observation['avg_service_time'])
        queue_len = np.array(observation['queue_length'])
        eligible = completion * (avg_service_time + queue_len)
        return np.argmax(eligible)

    @staticmethod
    def SEWMS(observation):
        """最短期望等待时间+平均服务时间优先"""
        completion = np.array(observation['completion'])
        avg_service_time = np.array(observation['avg_service_time'])
        queue_len = np.array(observation['queue_length'])
        completion = np.array(list(map(lambda x: 99999 if x == 0 else x, completion)))
        eligible = completion * (avg_service_time + queue_len)
        return np.argmin(eligible)

    @staticmethod
    def MW(observation):
        """最大繁忙程度优先"""
        completion = np.array(observation['completion'])
        workload = np.array(list(observation['work_load'].values())) + 1
        eligible = completion * workload
        return np.argmax(eligible)
