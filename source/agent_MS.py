import numpy as np
from numpy.random import Generator, PCG64


class MS(object):
    # Sticky Track-and-Stop
    def __init__(
        self, K: int, delta: float = 0.1, xi: float = 0.5, sigma_2: float = 1.0, random_seed: int = 12345
    ) -> None:
        # we use the prior as N(1, sigma_2)
        self.delta = delta
        self.K = K
        self.xi = xi
        self.sigma_2 = sigma_2
        self.sigma = np.sqrt(sigma_2)
        self.random_seed = random_seed
        self.random_generator = Generator(PCG64(random_seed))

        self.mean_reward_ = np.zeros(K)
        self.sum_pulling_fraction = np.zeros(K)
        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.action_ = list()
        self.t = 1

        self.pulling_list = [kk for kk in range(1, K + 1)]

        self.W_inverse = lambda x: x + np.log(x)
        self.c_t_delta = (
            lambda x: 1
            / 2
            * self.sigma_2
            * self.W_inverse(2 * np.log(self.K / self.delta) + 4 * np.log(np.log(x) + 4) + 1 / 2)
        )

        self.stop = False
        self.count_get_it = 0

    def action(self):
        assert not self.stop, "the algorithm stops"
        assert len(self.pulling_list) > 0, "pulling list is empty"

        arm = self.pulling_list.pop(0)
        self.action_.append(arm)
        return arm

    def observe(self, reward):
        assert not self.stop, "the algorithm stops"
        arm = self.action_[self.t - 1]
        self.total_reward_[arm - 1] += reward
        self.pulling_times_[arm - 1] += 1
        self.mean_reward_[arm - 1] = self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
        self.t += 1

        # calculate the arm to be pulled in the next round
        if len(self.pulling_list) == 0:
            # we imitate the sampling rule in Jourdan 2023
            # keep sampling until the sampling mean vector is positive
            post_mean = self.total_reward_ / self.pulling_times_
            posterior = (
                self.random_generator.normal(loc=0, scale=1, size=self.K) * (self.sigma / np.sqrt(self.pulling_times_))
                + post_mean
            )
            while np.max(posterior) < self.xi:
                posterior = (
                    self.random_generator.normal(loc=0, scale=1, size=self.K)
                    * (self.sigma / np.sqrt(self.pulling_times_))
                    + post_mean
                )
            arm = np.argmax(posterior) + 1
            self.pulling_list.append(arm)

        # determine whether to stop
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

    def if_stop(self):
        return self.stop


# %% unit test 1, test MS
# from env import Environment_Gaussian
# from tqdm import tqdm

# K = 30
# delta = 0.001
# mu0 = 0.5
# mu1 = 0.65
# n_exp = 100

# rlist = np.ones(K) * mu0
# rlist[0 : K // 4] = mu1

# stop_time_ = np.zeros(n_exp)
# correctness_ = np.ones(n_exp)
# phase_num_ = np.zeros(n_exp)
# for exp_id in tqdm(range(n_exp)):
#     np.random.seed(exp_id)
#     new_random_seed = np.random.randint(low=0, high=999999999)
#     np.random.seed(new_random_seed)
#     rlist_temp = rlist.copy()
#     np.random.shuffle(rlist_temp)
#     answer_set = list(np.where(rlist_temp > mu0)[0] + 1)

#     env = Environment_Gaussian(rlist=rlist_temp, K=K, random_seed=new_random_seed)
#     agent = MS(K=K, delta=delta, xi=mu0)

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
# mean_success = np.mean(correctness_)
# print("mean stop", mean_stop_time, "success rate", mean_success)
