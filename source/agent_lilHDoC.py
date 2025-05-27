import numpy as np


def bisection_e(r, v, tol=1e-6):
    # r is a monotonely increasing function of epsilon
    # this function wants to find the maximum epsilon that make r is less or equal to v
    a = 0
    b = 1
    while True:
        epsilon = (a + b) / 2
        fe = r(epsilon)
        if fe <= v:
            a = epsilon
        else:
            b = epsilon
        if fe <= v and v - fe < tol:
            return epsilon


def bisection_T(fun_T, v, ub, tol=1e-6):
    # fun_T is a monotonely decreasing function of T
    # this function wants to find the minimum T that make fun_T is no less than v
    a = 1
    b = ub
    while True:
        t = (a + b) / 2
        ft = fun_T(t)
        if ft <= v:
            a = t
        else:
            b = t
        if ft >= v and ft - v < tol:
            return t


class lilHDoC(object):
    def __init__(self, K: int, delta: float = 0.1, xi: float = 0.5, sigma_2: float = 1, T: int = 200) -> None:
        """_summary_

        Args:
            K (int): Arm number
            delta (float, optional): Tolerance level of error. Defaults to 0.1.
            xi (float, optional): The threshold. Defaults to 0.5.
            sigma_2 (float, optional): Subgaussian proxy. Defaults to 1.
            T (int, optional): The length of warm up stage. Defaults to 200.
        """

        assert delta > 0.0 and delta < 1.0, "delta is not in (0, 1)"

        self.delta = delta
        self.K = K
        self.xi = xi
        self.sigma_2 = sigma_2

        self.B = K + 1
        self.C = np.maximum(1 / delta, np.exp(1))

        # determine epsilon and r
        temp_B_C = np.minimum(np.log(np.log(self.B)) / np.log(self.B), np.log(np.log(self.C)) / np.log(self.C)) + 1
        fun_r = lambda e: ((1 + np.sqrt(e)) ** 2) * (1 + e)
        # use bisection to determine epsilon
        self.epsilon = bisection_e(fun_r, temp_B_C + 1)
        self.r = ((1 + np.sqrt(self.epsilon)) ** 2) * (1 + self.epsilon)
        self.c_epsilon = (2 + self.epsilon) / self.epsilon * (1 / np.log(1 + self.epsilon)) ** (1 + self.epsilon)

        # determine T
        if T is not None:
            self.T = T
        else:
            # we use the default way to calculate T
            temp_c_K_delta = 1 / 4 * K ** (self.r - 1) * (1 / self.delta) ** (self.r - 1) * (self.c_epsilon) ** (self.r)
            ub_T = K ** (self.r) * (1 / self.delta) ** (self.r) * (self.c_epsilon) ** (self.r)
            fun_T = lambda t: t**2 / (np.log((1 + self.epsilon) * t)) ** self.r
            self.T = bisection_T(fun_T, temp_c_K_delta, ub_T)
            self.T = np.floor(self.T)

        self.mean_reward_ = np.zeros(K)
        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.action_ = list()
        self.t = 1

        self.survive_arms = np.arange(1, K + 1)
        self.pulling_list = [kk for kk in range(1, K + 1)]
        self.pulling_list = self.pulling_list * int(self.T)

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

        if np.min(self.pulling_times_) < self.T:
            return None

        # check whether we should eliminate, or output this arm
        output_arm = None
        prob_conf_len = self.U_highprob(Nt=self.pulling_times_[arm - 1], w=self.delta / self.c_epsilon / self.K)
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
        U_t = np.sqrt(2 * self.sigma_2 * np.log(self.t) / Nt)
        return U_t

    def U_highprob(self, Nt, w):
        # bernoulli stopping rule
        # U_t_delta = np.sqrt(np.log(4 * self.K * (Nt**2) / self.delta) / 2 / Nt)

        # Gaussian stopping rule
        # U_t_delta = np.sqrt(2 * np.log(4 * self.K * (Nt**2) / self.delta) / Nt)

        U_t_delta = (1 + np.sqrt(self.epsilon)) * np.sqrt(
            2 * self.sigma_2 * (1 + self.epsilon) / Nt * np.log(np.log(Nt * (1 + self.epsilon)) / w)
        )
        return U_t_delta

    def if_stop(self):
        return self.stop


# %% unit test 1, test the bisection_e  and bisection_T
# K = 10
# delta = 0.01

# B = K + 1
# C = np.maximum(1 / delta, np.exp(1))

# # determine epsilon and r
# temp_B_C = np.minimum(np.log(np.log(B)) / np.log(B), np.log(np.log(C)) / np.log(C)) + 1
# fun_r = lambda e: ((1 + np.sqrt(e)) ** 2) * (1 + e)
# epsilon = bisection_e(fun_r, temp_B_C)
# r = ((1 + np.sqrt(epsilon)) ** 2) * (1 + epsilon)
# print(np.log(np.log(B)) / np.log(B))
# print(np.log(np.log(C)) / np.log(C))
# fun_r = lambda e: ((1 + np.sqrt(e)) ** 2) * (1 + e)
# epsilon = bisection_e(fun_r, temp_B_C)
# print(epsilon)
# print("approximated value of fun_r", fun_r(epsilon))
# print("threshold", temp_B_C)

# # determine T
# c_epsilon = (2 + epsilon) / epsilon * (1 / np.log(1 + epsilon)) ** (1 + epsilon)
# temp_c_K_delta = 1 / 4 * K ** (r - 1) * (1 / delta) ** (r - 1) * (c_epsilon) ** (r)
# ub_T = K ** (r) * (1 / delta) ** (r) * (c_epsilon) ** (r)
# fun_T = lambda t: t**2 / (np.log((1 + epsilon) * t)) ** r
# T = bisection_T(fun_T, temp_c_K_delta, ub_T)
# print(T)
# print("approximated value of fun_r", fun_T(T))
# print("threshold", temp_c_K_delta)

# %% unit test 2, test lilHDoC
# in this experiment, we need to set sigma^2=1/4
# from env import Environment_Gaussian, Environment_Bernoulli
# from tqdm import tqdm

# # K = 10
# # xi = 0.5
# # Delta = 0.15
# # rlist = np.ones(K) * xi
# # # rlist[1 : (K + 1) // 2] = xi + Delta
# # # rlist[(K + 1) // 2 : K] = xi - Delta
# # # rlist[0] = 1.0
# # rlist[0] = xi + Delta

# # rlist = np.array([0.1, 0.2, 0.7, 0.8])
# # K = len(rlist)
# # xi = 0.5
# # delta = 0.1

# rlist = np.array([0.007, 0.006, 0.005, 0.003, 0.002, 0.001])
# K = len(rlist)
# xi = 0.004

# delta = 0.01
# n_exp = 10

# for alg_class in [lilHDoC]:
#     stop_time_ = np.zeros(n_exp)
#     output_arm_ = list()
#     correctness_ = np.ones(n_exp)
#     for exp_id in tqdm(range(n_exp)):
#         rlist_temp = rlist.copy()
#         np.random.seed(exp_id)
#         np.random.shuffle(rlist_temp)
#         answer_set = list(np.where(rlist_temp > xi)[0] + 1)

#         env = Environment_Bernoulli(rlist=rlist_temp, K=K, random_seed=exp_id)
#         # env = Environment_Gaussian(rlist=rlist_temp, K=K, random_seed=exp_id)
#         agent = alg_class(K=K, delta=delta, xi=xi, sigma_2=1 / 4, T=10)
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
