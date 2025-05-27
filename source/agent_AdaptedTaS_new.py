import numpy as np


class Adapted_TaS_new(object):
    # Sticky Track-and-Stop
    def __init__(
        self, K: int, delta: float = 0.1, xi: float = 0.5, sigma_2: float = 1.0, logC=None, log1_over_delta=None
    ) -> None:
        self.delta = delta
        self.K = K
        self.xi = xi
        self.sigma_2 = sigma_2

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
            ## C-Track
            epsilon = 1 / np.sqrt(self.K**2 + self.t)
            projected_w = self.get_projected_w(self.mean_reward_, epsilon)
            self.sum_pulling_fraction = self.sum_pulling_fraction + projected_w
            arm = np.argmax(self.sum_pulling_fraction - self.pulling_times_) + 1
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

    def get_projected_w(self, hatmu, epsilon):
        max_mean = np.max(hatmu)
        if max_mean < self.xi:
            wt = 2 / (hatmu - self.xi) ** 2
            wt = wt / np.sum(wt)
            wt_projected = self.get_projection(wt, epsilon)
        else:  # max_mean \geq self.xi
            it = np.argmax(hatmu) + 1
            wt_projected = np.ones(self.K) * epsilon
            wt_projected[it - 1] = 1 - (self.K - 1) * epsilon
        return wt_projected

    def get_projection(self, w, epsilon):
        sorted_index_w = np.argsort(w)
        projected_w = np.zeros(self.K)
        B = 0
        for j in range(self.K):
            arm_index = sorted_index_w[j] + 1
            if w[arm_index - 1] <= epsilon:
                projected_w[arm_index - 1] = epsilon
                B += epsilon - w[arm_index - 1]
            else:
                if B / (self.K - 1 - j + 1) <= (w[arm_index - 1] - epsilon):
                    projected_w[sorted_index_w[j:]] = w[sorted_index_w[j:]] - B / (self.K - 1 - j + 1)
                    return projected_w
                else:
                    projected_w[arm_index - 1] = epsilon
                    B -= w[arm_index - 1] - epsilon

        return projected_w

    def Get_wt(self, hatmu, pulling):
        max_mean = np.max(hatmu)
        if max_mean < self.xi:
            wt = 2 / (hatmu - self.xi) ** 2
            wt = wt / np.sum(wt)
            return wt
        else:  # max_mean \geq self.xi
            it = np.argmax(hatmu) + 1
            wt = np.zeros(self.K)
            wt[it - 1] = 1
            return wt
