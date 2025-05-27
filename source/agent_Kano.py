import numpy as np


class HDoC_Kano(object):
    def __init__(self, K: int, delta: float = 0.1, xi: float = 0.5, sigma_2: float = 1.0) -> None:
        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"

        self.delta = delta
        self.K = K
        self.xi = xi
        self.sigma_2 = sigma_2

        self.mean_reward_ = np.zeros(K)
        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.action_ = list()
        self.t = 1

        self.survive_arms = np.arange(1, K + 1)
        self.pulling_list = [kk for kk in range(1, K + 1)]

        self.stop = False

    def action(self):
        assert not self.stop, "the algorithm stops"
        assert len(self.pulling_list) > 0, "pulling list is empty"
        assert len(self.survive_arms) >= 1, "the algorithm stops"

        arm = self.pulling_list.pop(0)
        self.action_.append(arm)
        return arm

    def observe(self, reward):
        assert len(self.survive_arms) >= 1, "the algorithm stops"
        arm = self.action_[self.t - 1]
        self.total_reward_[arm - 1] += reward
        self.pulling_times_[arm - 1] += 1
        self.mean_reward_[arm - 1] = self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
        self.t += 1

        # check whether we should eliminate, or output this arm
        output_arm = None
        prob_conf_len = self.U_highprob(Nt=self.pulling_times_[arm - 1])
        prob_upper = self.mean_reward_[arm - 1] + prob_conf_len
        prob_lower = self.mean_reward_[arm - 1] - prob_conf_len
        if prob_upper < self.xi:
            self.survive_arms = self.survive_arms[self.survive_arms != arm]
            if len(self.survive_arms) == 0:
                self.stop = True
                return "No Arms Above xi"
        if prob_lower >= self.xi:
            output_arm = arm
            self.stop = True
            return output_arm
            # self.survive_arms = self.survive_arms[self.survive_arms != arm]
            # if len(self.survive_arms) == 0:
            #     self.stop = True
            #     return output_arm

        if len(self.pulling_list) == 0:
            # determine the next arm to pull
            conf_len = self.U(Nt=self.pulling_times_[self.survive_arms - 1])
            upper_bound = self.mean_reward_[self.survive_arms - 1] + conf_len
            hat_a = self.survive_arms[np.argmax(upper_bound)]
            self.pulling_list.append(hat_a)

        return output_arm

    def U(self, Nt):
        # U_t = np.sqrt(np.log(self.t) / 2 / Nt)
        U_t = np.sqrt(2 * np.log(self.t) / Nt)
        return U_t

    def U_highprob(self, Nt):
        # bernoulli stopping rule
        # U_t_delta = np.sqrt(np.log(4 * self.K * (Nt**2) / self.delta) / 2 / Nt)

        # Gaussian stopping rule
        U_t_delta = np.sqrt(2 * self.sigma_2 * np.log(4 * self.K * (Nt**2) / self.delta) / Nt)
        return U_t_delta

    def if_stop(self):
        return self.stop


class LUCB_G_Kano(object):
    def __init__(self, K: int, delta: float = 0.1, xi: float = 0.5, sigma_2: float = 1.0) -> None:
        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"

        self.delta = delta
        self.K = K
        self.xi = xi
        self.sigma_2 = sigma_2

        self.mean_reward_ = np.zeros(K)
        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.action_ = list()
        self.t = 1

        self.survive_arms = np.arange(1, K + 1)
        self.pulling_list = [kk for kk in range(1, K + 1)]

        self.stop = False

    def action(self):
        assert not self.stop, "the algorithm stops"
        assert len(self.pulling_list) > 0, "pulling list is empty"
        assert len(self.survive_arms) >= 1, "the algorithm stops"

        arm = self.pulling_list.pop(0)
        self.action_.append(arm)
        return arm

    def observe(self, reward):
        assert len(self.survive_arms) >= 1, "the algorithm stops"
        arm = self.action_[self.t - 1]
        self.total_reward_[arm - 1] += reward
        self.pulling_times_[arm - 1] += 1
        self.mean_reward_[arm - 1] = self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
        self.t += 1

        # check whether we should eliminate, or output this arm
        output_arm = None
        prob_conf_len = self.U_highprob(Nt=self.pulling_times_[arm - 1])
        prob_upper = self.mean_reward_[arm - 1] + prob_conf_len
        prob_lower = self.mean_reward_[arm - 1] - prob_conf_len
        if prob_upper < self.xi:
            self.survive_arms = self.survive_arms[self.survive_arms != arm]
            if len(self.survive_arms) == 0:
                self.stop = True
                return "No Arms Above xi"
        if prob_lower >= self.xi:
            output_arm = arm
            self.stop = True
            return output_arm

        if len(self.pulling_list) == 0:
            # determine the arm to pull
            conf_len = self.U_highprob(Nt=self.pulling_times_[self.survive_arms - 1])
            prob_upper = self.mean_reward_[self.survive_arms - 1] + conf_len
            hat_a = self.survive_arms[np.argmax(prob_upper)]
            self.pulling_list.append(hat_a)

        return output_arm

    def U_highprob(self, Nt):
        # Bernoulli high prob
        # U_t_delta = np.sqrt(np.log(4 * self.K * (Nt**2) / self.delta) / 2 / Nt)

        # Gaussian high prob
        U_t_delta = np.sqrt(2 * self.sigma_2 * np.log(4 * self.K * (Nt**2) / self.delta) / Nt)
        return U_t_delta

    def if_stop(self):
        return self.stop


class APT_G_Kano(object):
    def __init__(self, K: int, delta: float = 0.1, xi: float = 0.5) -> None:
        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"

        self.delta = delta
        self.K = K
        self.xi = xi

        self.mean_reward_ = np.zeros(K)
        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.action_ = list()
        self.t = 1

        self.survive_arms = np.arange(1, K + 1)
        self.pulling_list = [kk for kk in range(1, K + 1)]

        self.stop = False

    def action(self):
        assert not self.stop, "the algorithm stops"
        assert len(self.pulling_list) > 0, "pulling list is empty"
        assert len(self.survive_arms) >= 1, "the algorithm stops"

        arm = self.pulling_list.pop(0)
        self.action_.append(arm)
        return arm

    def observe(self, reward):
        assert len(self.survive_arms) >= 1, "the algorithm stops"
        arm = self.action_[self.t - 1]
        self.total_reward_[arm - 1] += reward
        self.pulling_times_[arm - 1] += 1
        self.mean_reward_[arm - 1] = self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
        self.t += 1

        # check whether we should eliminate, or output this arm
        output_arm = None
        prob_conf_len = self.U_highprob(Nt=self.pulling_times_[arm - 1])
        prob_upper = self.mean_reward_[arm - 1] + prob_conf_len
        prob_lower = self.mean_reward_[arm - 1] - prob_conf_len
        if prob_upper < self.xi:
            self.survive_arms = self.survive_arms[self.survive_arms != arm]
            if len(self.survive_arms) == 0:
                self.stop = True
                return output_arm
        if prob_lower >= self.xi:
            output_arm = arm
            self.survive_arms = self.survive_arms[self.survive_arms != arm]
            if len(self.survive_arms) == 0:
                self.stop = True
                return output_arm

        if len(self.pulling_list) == 0:
            # determine the next arm to pull
            beta_t = np.sqrt(self.pulling_times_[self.survive_arms - 1]) * np.abs(
                self.mean_reward_[self.survive_arms - 1] - self.xi
            )
            hat_a = self.survive_arms[np.argmin(beta_t)]

            self.pulling_list.append(hat_a)

        return output_arm

    def U(self, Nt):
        U_t = np.sqrt(np.log(self.t) / 2 / Nt)
        return U_t

    def U_highprob(self, Nt):
        # U_t_delta = np.sqrt(np.log(4 * self.K * (Nt**2) / self.delta) / 2 / Nt)
        U_t_delta = np.sqrt(2 * np.log(4 * self.K * (Nt**2) / self.delta) / Nt)
        return U_t_delta

    def if_stop(self):
        return self.stop


# %% unit test, test HDoC_Kano, LUCB_G_Kano, APT_G_Kano
# from env import Environment_Gaussian
# from tqdm import tqdm

# # K = 11
# # xi = 0.5
# # Delta = 0.2
# # rlist = np.ones(K) * xi
# # rlist[1 : (K + 1) // 2] = xi + Delta
# # rlist[(K + 1) // 2 : K] = xi - Delta
# # # rlist[0] = 1.0
# # rlist[0] = 1.0

# rlist = np.array([0.1, 0.2, 0.7, 0.8])
# K = len(rlist)
# xi = 0.5
# delta = 0.1

# delta = 0.1
# n_exp = 1000

# for alg_class in [HDoC_Kano, LUCB_G_Kano, APT_G_Kano]:
#     # # for alg_class in [HDoC_Kano, LUCB_G_Kano]:
#     # # for alg_class in [HDoC_Kano]:
#     # # for alg_class in [LUCB_G_Kano]:
#     # for alg_class in [APT_G_Kano]:
#     stop_time_ = np.zeros(n_exp)
#     output_arm_ = list()
#     correctness_ = np.ones(n_exp)
#     for exp_id in tqdm(range(n_exp)):
#         rlist_temp = rlist.copy()
#         # np.random.seed(exp_id)
#         # np.random.shuffle(rlist_temp)
#         answer_set = list(np.where(rlist_temp > xi)[0] + 1)

#         # env = Environment_Bernoulli(rlist=rlist_temp, K=K, random_seed=exp_id)
#         env = Environment_Gaussian(rlist=rlist_temp, K=K, random_seed=exp_id)
#         agent = alg_class(K=K, delta=delta, xi=xi)
#         while not agent.stop:
#             arm = agent.action()
#             reward = env.response(arm)
#             output_arm = agent.observe(reward)
#             if output_arm is not None:
#                 output_arm_.append(output_arm)
#                 break
#         stop_time_[exp_id] = agent.t
#         if output_arm not in answer_set or output_arm == -1:
#             correctness_[exp_id] = 0
#     mean_stop_time = np.mean(stop_time_)
#     mean_success = np.mean(correctness_)
#     algname = type(agent).__name__
#     print(f"For algorithm {algname}, ")
#     print(f"mean stop time is {mean_stop_time}")
#     print(f"mean correctness rate is {mean_success}")

# %% unit test, test HDoC_Kano, LUCB_G_Kano for instance Th3
# from env import Environment_Bernoulli
# from tqdm import tqdm

# rlist = np.ones(10)
# K = len(rlist)
# rlist[0:3] = 0.55
# rlist[3:10] = 0.45
# print(rlist)

# xi = 0.5
# delta = 0.01

# n_exp = 1000
# np.random.seed(0)
# new_random_seed = np.random.randint(low=0, high=1000000)
# for alg_class in [HDoC_Kano]:
#     stop_time_ = np.zeros(n_exp)
#     output_arm_ = list()
#     correctness_ = np.ones(n_exp)
#     for exp_id in tqdm(range(n_exp)):
#         this_random_seed = exp_id * new_random_seed
#         rlist_temp = rlist.copy()
#         np.random.seed(this_random_seed)
#         np.random.shuffle(rlist_temp)
#         answer_set = list(np.where(rlist_temp > xi)[0] + 1)

#         env = Environment_Bernoulli(rlist=rlist_temp, K=K, random_seed=this_random_seed)
#         # env = Environment_Gaussian(rlist=rlist_temp, K=K, random_seed=this_random_seed)
#         agent = alg_class(K=K, delta=delta, xi=xi, sigma_2=1/4)
#         while not agent.stop:
#             arm = agent.action()
#             reward = env.response(arm)
#             output_arm = agent.observe(reward)
#             if output_arm is not None:
#                 output_arm_.append(output_arm)
#                 break
#         stop_time_[exp_id] = agent.t
#         if output_arm not in answer_set or output_arm == -1:
#             correctness_[exp_id] = 0
#     mean_stop_time = np.mean(stop_time_)
#     std_stop_time = np.std(stop_time_) / np.sqrt(n_exp)
#     mean_success = np.mean(correctness_)
#     algname = type(agent).__name__
#     print(f"For algorithm {algname}, ")
#     print(f"mean stop time is {mean_stop_time}, std {std_stop_time}")
#     print(f"mean correctness rate is {mean_success}")
