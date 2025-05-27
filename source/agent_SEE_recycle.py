import numpy as np


def U_conf(delta, Nt, sigma=1):
    Nt_log_ceil = np.maximum(np.ceil(np.log2(Nt)), 1)
    Nt_ceil_2 = 2**Nt_log_ceil

    U_t_delta = np.sqrt(2 * (sigma**2) * Nt_ceil_2 * np.log(2 * Nt_log_ceil**2 / delta)) / Nt
    return U_t_delta


class SEE_recycle_ee(object):
    def __init__(self, K: int, delta: float = 0.01, xi: float = 0.5, C: float = 1.01, T: int = 1, sigma_2: float = 1.0):
        assert delta > 0.0 and delta <= 1.0, "delta is not in (0, 1]"
        self.delta = delta
        self.K = K
        self.xi = xi
        self.C = C
        self.T = T
        self.sigma = np.sqrt(sigma_2)

        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.mean_reward_ = np.zeros(K)
        self.Q = []  # temporary container
        self.t = 0

        self.action_ = list()
        self.pulling_list = [kk for kk in range(1, K + 1)]

        # each time we call the exploration oracle, we need to initialize these two variables.
        self.hata = None
        self.pause = False

    def action(self):
        # determine the arm to pull in the next round
        if len(self.pulling_list) == 0:
            while True:
                prob_conf_len = U_conf(
                    delta=self.delta / self.K, Nt=np.maximum(self.pulling_times_, 1), sigma=self.sigma
                )
                UCB_ = self.mean_reward_ + prob_conf_len
                arm = np.argmax(UCB_) + 1

                record_exists = False
                for a, reward in self.Q:
                    if a == arm:
                        self.Q.remove((a, reward))
                        self.observe(reward=reward, arm=arm)
                        record_exists = True
                        break
                if not record_exists:
                    self.pulling_list.append(arm)
                    break
                if self.pause:
                    return None

        if self.pause:
            return None

        arm = self.pulling_list.pop(0)
        self.action_.append(arm)
        self.t += 1
        return arm

    def observe(self, reward, arm=None):
        if arm is None:
            arm = self.action_[-1]
        self.pulling_times_[arm - 1] += 1
        self.total_reward_[arm - 1] += reward
        self.mean_reward_[arm - 1] = self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]

        # determine whether to terminate
        if self.t >= self.T:
            self.hata = "Not Complete"
            self.pause = True
        prob_conf_len = U_conf(delta=self.delta / self.K, Nt=np.maximum(self.pulling_times_, 1), sigma=self.sigma)
        UCB_ = self.mean_reward_ + prob_conf_len
        LCB_ = self.mean_reward_ - self.C * prob_conf_len
        if np.all(UCB_ < self.xi):
            self.hata = "No Arms Above xi"
            self.pause = True
        if LCB_[arm - 1] > self.xi:
            self.Q.append((arm, reward))
            self.pulling_times_[arm - 1] -= 1
            self.total_reward_[arm - 1] -= reward
            if self.pulling_times_[arm - 1] > 0:
                self.mean_reward_[arm - 1] = self.total_reward_[arm - 1] / self.pulling_times_[arm - 1]
            else:
                self.mean_reward_[arm - 1] = 0

            self.hata = arm
            self.pause = True


class SEE_recycle_et(object):
    def __init__(
        self, K: int, hata: int, alpha: float, delta: float = 0.01, xi: float = 0.5, T: int = 1, sigma_2: float = 1.0
    ):
        assert delta > 0.0 and delta <= 1.0, "delta is not in (0, 1]"
        assert (alpha >= 1 and alpha <= K) or hata is None, "hata is not in set [K]"
        self.delta = delta
        self.hata = hata
        self.alpha = alpha
        self.K = K
        self.xi = xi
        self.T = T
        self.sigma = np.sqrt(sigma_2)

        self.pulling_times_ = np.zeros(K)
        self.total_reward_ = np.zeros(K)
        self.mean_reward_ = np.zeros(K)
        self.t = 0

        self.action_ = list()

    def action(self):
        self.t += 1
        return self.hata

    def observe(self, reward):
        self.pulling_times_[self.hata - 1] += 1
        self.total_reward_[self.hata - 1] += reward
        self.mean_reward_[self.hata - 1] = self.total_reward_[self.hata - 1] / self.pulling_times_[self.hata - 1]

        prob_conf_len_hata = U_conf(
            delta=self.delta / self.K / self.alpha, Nt=self.pulling_times_[self.hata - 1], sigma=self.sigma
        )
        LCB_hata = self.mean_reward_[self.hata - 1] - prob_conf_len_hata
        if LCB_hata > self.xi:
            return "Qualified"
        if self.pulling_times_[self.hata - 1] >= self.T:
            self.hata = "Not Complete"
            self.pause = True
            return self.hata
        return None


class SEE_recycle(object):
    def __init__(
        self,
        K: int,
        delta_k_,
        alpha_k_,
        beta_k_,
        delta: float = 0.01,
        xi: float = 0.5,
        C: float = 1.01,
        delta_fraction_C: float = 3.0,
        sigma_2: float = 1.0,
    ):
        assert delta > 0.0 and delta <= 1.0, "delta is not in (0, 1]"
        self.delta = delta
        self.K = K
        self.xi = xi
        self.delta_k_ = delta_k_
        self.beta_k_ = beta_k_
        self.alpha_k_ = alpha_k_
        self.C = C
        self.delta_fraction_C = delta_fraction_C
        self.sigma_2 = sigma_2

        # self.pulling_times_ee_ = np.zeros(K)
        # self.pulling_times_et_ = np.zeros(K)
        # self.total_reward_ee_ = np.zeros(K)
        # self.total_reward_et_ = np.zeros(K)
        # self.mean_reward_ee_ = np.zeros(K)
        # self.mean_reward_et_ = np.zeros(K)
        # self.Q = []  # temporary container

        self.action_ = list()

        self.phase_index = 1
        self.delta_k = delta_k_(1)
        self.beta_k = beta_k_(1)
        self.alpha_k = alpha_k_(1)

        self.period = "explore"  # it can take value "explore" or "exploit"
        self.hat_a_ = list()  # all the history output arm from exploration period
        self.hat_a = 0  # current hat_a to be explored

        self.ee_oracle = SEE_recycle_ee(
            K=self.K,
            delta=self.delta_k,
            xi=self.xi,
            C=self.C,
            T=1000 * ((C + 1) ** 2) * K * self.beta_k * np.log(4 * K / self.delta_k),
            sigma_2=sigma_2,
        )
        self.et_oracle = SEE_recycle_et(
            K=K,
            hata=None,
            alpha=self.alpha_k,
            delta=delta,
            xi=xi,
            T=1,
            sigma_2=sigma_2,
        )

        self.t = 0
        self.stop = False

    def action(self):
        assert not self.stop, "the algorithm stops"
        assert self.period == "explore" or self.period == "exploit", "unknown phase period"

        if self.period == "explore":
            arm = self.ee_oracle.action()
            if self.ee_oracle.pause and arm is None:
                # this case only happens after we transfer a record (a, X) from Q to H^{ee},
                # and then one of the stopping conditions is triggered
                # the stopping condition cannot be $t>=T$, as transfering (a, X) from Q to H^{ee} will not increase t
                # the stopping condition cannot be $UCB(a')<=\xi,\foral a'$, the reason is
                #     when $(a, X)$ was transferred from H^{ee} to Q, $UCB(a)>LCB(a)>\xi$
                #     then at a new exploration period, $UCB(a)$ will get larger
                # then we can conclude the stopping rule $LCB(a)>\xi$ is triggered
                assert 1 <= self.ee_oracle.hata and self.ee_oracle.hata <= self.K, "Algorithm logic breaks down"
                self.period = "exploit"
                self.hat_a = self.ee_oracle.hata
                self.hat_a_.append(self.ee_oracle.hata)

                self.alpha_k = self.alpha_k_(self.phase_index)
                self.beta_k = self.beta_k_(self.phase_index)
                T_k_et = 1000 * self.sigma_2 * self.beta_k * np.log(4 * self.alpha_k * self.K / self.delta)

                self.et_oracle.hata = self.ee_oracle.hata
                self.et_oracle.alpha = self.alpha_k
                self.T = T_k_et

                arm = self.et_oracle.action()
                self.action_.append(arm)
                self.t += 1
                return arm
            else:
                # in this case, we are still in exploration period
                # the explore oracle will return arm in [K]
                self.action_.append(arm)
                self.t += 1
                return arm
        elif self.period == "exploit":  # exploitation
            arm = self.et_oracle.action()
            self.action_.append(arm)
            self.t += 1
            return arm

    def observe(self, reward):
        assert not self.stop, "the algorithm stops"
        assert self.period == "explore" or self.period == "exploit", "unknown phase period"
        if self.period == "explore":
            self.ee_oracle.observe(reward=reward)
            if self.ee_oracle.pause:
                # then we determine the next step based on self.ee_oracle.hata
                if self.ee_oracle.hata == "Not Complete":
                    self.period = "explore"
                    self.phase_index += 1

                    self.beta_k = self.beta_k_(self.phase_index)
                    self.delta_k = self.delta_k_(self.phase_index)
                    T_k_ee = (
                        1000
                        * self.sigma_2
                        * ((self.C + 1) ** 2)
                        * self.K
                        * self.beta_k
                        * np.log(4 * self.K / self.delta_k)
                    )

                    self.ee_oracle.hata = None
                    self.ee_oracle.pause = False
                    self.ee_oracle.T = T_k_ee
                    self.ee_oracle.delta = self.delta_k
                elif self.ee_oracle.hata == "No Arms Above xi" and self.delta_k >= self.delta / self.delta_fraction_C:
                    self.period = "explore"
                    self.phase_index += 1

                    self.beta_k = self.beta_k_(self.phase_index)
                    self.delta_k = self.delta_k_(self.phase_index)
                    T_k_ee = (
                        1000
                        * self.sigma_2
                        * ((self.C + 1) ** 2)
                        * self.K
                        * self.beta_k
                        * np.log(4 * self.K / self.delta_k)
                    )

                    self.ee_oracle.hata = None
                    self.ee_oracle.pause = False
                    self.ee_oracle.T = T_k_ee
                    self.ee_oracle.delta = self.delta_k
                elif self.ee_oracle.hata == "No Arms Above xi" and self.delta_k < self.delta / self.delta_fraction_C:
                    self.hat_a = "No Arms Above xi"
                    self.stop = True
                    return self.hat_a
                elif 1 <= self.ee_oracle.hata <= self.K:
                    self.period = "exploit"
                    self.hat_a = self.ee_oracle.hata
                    self.hat_a_.append(self.ee_oracle.hata)

                    self.alpha_k = self.alpha_k_(self.phase_index)
                    self.beta_k = self.beta_k_(self.phase_index)
                    T_k_et = 1000 * self.sigma_2 * self.beta_k * np.log(4 * self.alpha_k * self.K / self.delta)

                    self.et_oracle.hata = self.ee_oracle.hata
                    self.et_oracle.alpha = self.alpha_k
                    self.et_oracle.T = T_k_et
        elif self.period == "exploit":
            result = self.et_oracle.observe(reward=reward)
            if result == "Qualified":
                self.stop = True
                return self.hat_a
            elif result == "Not Complete":
                self.period = "explore"
                self.phase_index += 1

                self.beta_k = self.beta_k_(self.phase_index)
                self.delta_k = self.delta_k_(self.phase_index)
                T_k_ee = (
                    1000 * self.sigma_2 * ((self.C + 1) ** 2) * self.K * self.beta_k * np.log(4 * self.K / self.delta_k)
                )

                self.ee_oracle.hata = None
                self.ee_oracle.pause = False
                self.ee_oracle.T = T_k_ee
                self.ee_oracle.delta = self.delta_k


# %% unit test 1, test SEE_recycle_ee
# from env import Environment_Gaussian
# from tqdm import tqdm

# K = 10
# delta = 0.01
# mu0 = 0.5
# mu1 = 0.65
# n_exp = 100
# debug_test = True

# rlist = np.ones(K) * mu0
# rlist[-1] = mu1

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
#     agent = SEE_recycle_ee(K=K, delta=delta, xi=mu0, T=100000)
#     while not agent.pause:
#         arm = agent.action()
#         if arm is not None:
#             reward = env.response(arm)
#             if debug_test and agent.t >= K + 1:
#                 reward = 100
#             agent.observe(reward)

#     stop_time_[exp_id] = agent.t
#     if agent.hata not in answer_set:
#         correctness_[exp_id] = 0

#     agent.delta = delta / 10
#     agent.hata = None
#     agent.pause = False
#     while not agent.pause:
#         arm = agent.action()
#         if arm is not None:
#             reward = env.response(arm)
#             agent.observe(reward)
#     if agent.hata not in answer_set:
#         correctness_[exp_id] = 0


# mean_stop_time = np.mean(stop_time_)
# mean_success = np.mean(correctness_)
# print("mean stop", mean_stop_time, "success rate", mean_success)

# %% unit test 2, test SEE_recycle, Unique Qualified
# from env import Environment_Gaussian
# from tqdm import tqdm

# K = 10
# delta = 0.001
# mu0 = 0.5
# mu1 = 0.75
# n_exp = 100

# rlist = np.ones(K) * mu0
# rlist[-1] = mu1

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

#     delta_k_ = lambda x: 1 / 3**x
#     beta_k_ = lambda x: 2**x
#     alpha_k_ = lambda x: 5**x

#     agent = SEE_recycle(K=K, delta_k_=delta_k_, alpha_k_=alpha_k_, beta_k_=beta_k_, delta=delta, xi=mu0, C=1.01)

#     while not agent.stop:
#         arm = agent.action()
#         reward = env.response(arm)
#         output_arm = agent.observe(reward)
#         if output_arm is not None:
#             # print("total rounds", agent.t, "phase num", agent.phase_index)
#             break
#     stop_time_[exp_id] = agent.t
#     phase_num_[exp_id] = agent.phase_index
#     if output_arm not in answer_set:
#         correctness_[exp_id] = 0
# mean_stop_time = np.mean(stop_time_)
# mean_success = np.mean(correctness_)
# print("mean stop", mean_stop_time, "success rate", mean_success)

# %% unit test 3, all below mu0
# from env import Environment_Gaussian
# from tqdm import tqdm

# K = 10
# delta = 0.01
# mu0 = 0.5
# Delta = 0.25
# mu1 = 0.75
# n_exp = 100

# rlist = np.ones(K) * (mu0 - Delta)

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
#     delta_k_ = lambda x: 1 / 3**x
#     beta_k_ = lambda x: 2**x
#     alpha_k_ = lambda x: 5**x
#     agent = SEE_recycle(K=K, delta_k_=delta_k_, alpha_k_=alpha_k_, beta_k_=beta_k_, delta=delta, xi=mu0, C=1.01)

#     while not agent.stop:
#         arm = agent.action()
#         reward = env.response(arm)
#         output_arm = agent.observe(reward)
#         if output_arm is not None:
#             # print("total rounds", agent.t, "phase num", agent.phase_index)
#             break
#     stop_time_[exp_id] = agent.t
#     phase_num_[exp_id] = agent.phase_index
#     if output_arm != "No Arms Above xi":
#         correctness_[exp_id] = 0
# mean_stop_time = np.mean(stop_time_)
# std_stop_time = np.std(stop_time_) / np.sqrt(n_exp)
# mean_success = np.mean(correctness_)
# print("mean stop", mean_stop_time, "std", std_stop_time, "success rate", mean_success)
