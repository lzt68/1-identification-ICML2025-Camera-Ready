import numpy as np


class APGAI(object):
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

        self.W_inverse = lambda x: x + np.log(x)
        self.c_t_delta = (
            lambda x: 1
            / 2
            * self.sigma_2
            * self.W_inverse(2 * np.log(self.K / self.delta) + 4 * np.log(np.log(x) + 4) + 1 / 2)
        )

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

        # check whether we should stop
        cond = np.sqrt(2 * self.c_t_delta(self.t - 1))

        W_plus = np.sqrt(self.pulling_times_) * np.maximum(self.mean_reward_ - self.xi, 0)
        arm_W_plus = np.argmax(W_plus) + 1
        if W_plus[arm_W_plus - 1] > cond:
            self.stop = True
            return arm_W_plus
        W_minus = np.sqrt(self.pulling_times_) * np.maximum(self.xi - self.mean_reward_, 0)
        arm_W_minus = np.argmin(W_minus) + 1
        if W_minus[arm_W_minus - 1] > cond:
            self.stop = True
            return "No Arms Above xi"

        if len(self.pulling_list) == 0:
            # determine the next arm to pull
            arm_max_emprical = np.argmax(self.mean_reward_) + 1
            if self.mean_reward_[arm_max_emprical - 1] > self.xi:
                self.pulling_list.append(arm_W_plus)
            else:
                self.pulling_list.append(arm_W_minus)

        return None

    def if_stop(self):
        return self.stop


# %% unit test, test APGAI
# from env import Environment_Gaussian

# rlist = [0.1, 0.2, 0.7, 0.8]
# K = len(rlist)
# xi = 0.5
# delta = 0.1

# env = Environment_Gaussian(rlist=rlist, K=K, random_seed=12345)
# # agent = AlgOracle(alg_handle=AdaptedAE, K=K, delta=delta, xi=xi)
# agent = APGAI(K=K, delta=delta, xi=xi)
# output_arm = None
# stop_time = 0
# while not agent.stop:
#     arm = agent.action()
#     reward = env.response(arm)
#     output_arm = agent.observe(reward)
#     if output_arm is not None:
#         predicted_arm = output_arm
#         stop_time = agent.t
# print(f"output arm is {output_arm}, output time is {stop_time}")

# %% unit test, test APGAI
# from env import Environment_Gaussian
# from tqdm import tqdm

# K = 10
# delta = 0.01
# mu0 = 0.5
# Delta = 0.05
# n_exp = 20

# rlist = np.linspace(start=mu0 - Delta, stop=mu0 + Delta, num=K)
# stop_time_ = np.zeros(n_exp)
# correctness_ = np.ones(n_exp)
# for exp_id in tqdm(range(n_exp)):
#     exp_id = 18

#     np.random.seed(exp_id)
#     new_random_seed = np.random.randint(low=0, high=999999999)
#     np.random.seed(new_random_seed)
#     rlist_temp = rlist.copy()
#     answer_set = list(np.where(rlist_temp > mu0)[0] + 1)

#     env = Environment_Gaussian(rlist=rlist_temp, K=K, random_seed=new_random_seed)
#     agent = APGAI(K=K, delta=delta, xi=mu0)
#     while not agent.stop:
#         arm = agent.action()
#         reward = env.response(arm)
#         output_arm = agent.observe(reward)
#         if output_arm is not None:
#             # print("total rounds", agent.t, "phase num", agent.phase_index)
#             break
#     stop_time_[exp_id] = agent.t
#     if output_arm not in answer_set:
#         correctness_[exp_id] = 0
# mean_stop_time = np.mean(stop_time_)
# std_stop_time = np.std(stop_time_) / np.sqrt(n_exp)
# mean_success = np.mean(correctness_)
# print(
#     "mean stop", mean_stop_time, "std stop", std_stop_time, "success rate", mean_success
# )


# %% unit test, test APGAI
# from env import Environment_Gaussian
# from tqdm import tqdm

# K = 100
# delta = 0.01
# mu0 = 0.5
# Delta = 0.05
# n_exp = 1000

# rlist = np.linspace(start=mu0 - Delta, stop=mu0 + Delta, num=K)
# stop_time_ = np.zeros(n_exp)
# correctness_ = np.ones(n_exp)
# for exp_id in tqdm(range(n_exp)):
#     np.random.seed(exp_id)
#     new_random_seed = np.random.randint(low=0, high=999999999)
#     np.random.seed(new_random_seed)
#     rlist_temp = rlist.copy()
#     answer_set = list(np.where(rlist_temp > mu0)[0] + 1)

#     env = Environment_Gaussian(rlist=rlist_temp, K=K, random_seed=new_random_seed)
#     agent = APGAI(K=K, delta=delta, xi=mu0)
#     while not agent.stop:
#         arm = agent.action()
#         reward = env.response(arm)
#         output_arm = agent.observe(reward)
#         if output_arm is not None:
#             # print("total rounds", agent.t, "phase num", agent.phase_index)
#             break
#     stop_time_[exp_id] = agent.t
#     if output_arm not in answer_set:
#         correctness_[exp_id] = 0
# mean_stop_time = np.mean(stop_time_)
# std_stop_time = np.std(stop_time_) / np.sqrt(n_exp)
# mean_success = np.mean(correctness_)
# print(
#     "mean stop", mean_stop_time, "std stop", std_stop_time, "success rate", mean_success
# )
# """
# mean stop 77553.291 std stop 11742.566451907789 success rate 1.0
# """
