from typing import Callable
import numpy as np
from tqdm import tqdm
import inspect


def Unique_Qualified(K, mu0=0.5, Delta=0.25):
    mu1 = mu0 + Delta
    rlist = np.ones(K) * mu0
    rlist[-1] = mu1
    return rlist


# def Unique_Good_other_bad(K, mu0=0.5, Delta=0.15):
#     mu1 = mu0 + Delta
#     rlist = np.zeros(K) * mu0
#     rlist[-1] = mu1
#     return rlist


# def Unique_Qualified_Small_Gap(K, mu0=0.5, Delta=0.05):
#     mu1 = mu0 + Delta
#     rlist = np.zeros(K) * mu0
#     rlist[-1] = mu1
#     return rlist


def OneQuarter_Qualified(K, mu0=0.5, Delta=0.25):
    mu1 = mu0 + Delta
    rlist = np.ones(K) * mu0
    rlist[0 : K // 4] = mu1
    return rlist


def Linear(K, mu0=0.5, Delta=0.25):
    rlist = np.linspace(start=mu0 - Delta, stop=mu0 + Delta, num=K)
    return rlist


# def Linearp2(K, mu0=0.5, Delta=0.25):
#     rlist = np.linspace(start=mu0 - Delta, stop=mu0 + Delta, num=K)
#     return rlist


def AllWorse(K, mu0=0.5, Delta=0.25):
    rlist = np.ones(K) * (mu0 - Delta)
    return rlist


def AllGood(K, mu0=0.5, Delta=0.25):
    rlist = np.ones(K) * (mu0 + Delta)
    return rlist


def HalfGood(K, mu0=0.5, Delta=0.25):
    mu1 = mu0 + Delta
    rlist = np.ones(K) * mu0
    rlist[0 : K // 2] = mu1
    return rlist


# def AllGood2(K, mu0=0.5, Delta=0.2):
#     rlist = np.ones(K) * (mu0 + Delta)
#     return rlist


def Experiment(
    rlist: np.ndarray,
    delta: float,
    K: int,
    xi: float,
    env_class: Callable,
    agent_class: Callable,
    random_seed_for_start: int = 0,
    disable_tqdm: bool = False,
    shuffle: bool = True,
    stop_benchmark: int = 10**8,
    n_exp: int = 1000,
):
    """Experiment Oracle

    Args:
        rlist (np.ndarray): reward vector of each arm
        delta (float): confidence level
        K (int): number of arms
        xi (float): the threshold
        env_class (Callable): the environment type
        agent_class (Callable): the 1-identification algorithm
        random_seed_for_start (int, optional): random seed for the experiment oracle. Defaults to 0.
        disable_tqdm (bool, optional): whether output tqdm info. Defaults to False.
        shuffle (bool, optional): whether shuffle the arm. Defaults to True.
        stop_benchmark (int, optional): the maximum length of an experiment. Defaults to 10**8.
        n_exp (int, optional): number of independent experiments. Defaults to 1000.
    """
    assert len(rlist) == K, "number of resources doesn't match"

    np.random.seed(random_seed_for_start)
    random_seed = np.random.randint(low=n_exp, high=n_exp * 100)
    stop_time_ = np.zeros(n_exp)
    correctness_ = np.ones(n_exp)
    exceed_stop_ = np.zeros(n_exp)
    for exp_id in tqdm(range(n_exp), disable=disable_tqdm):
        # shuffle the arm order
        rlist_temp = rlist.copy()
        new_random_seed = random_seed * exp_id
        if shuffle:
            np.random.seed(new_random_seed)
            np.random.shuffle(rlist_temp)
        answer_set = list(np.where(rlist_temp > xi)[0] + 1)
        if len(answer_set) == 0:
            answer_set = ["No Arms Above xi"]

        # define the environment and the agent
        env = env_class(rlist=rlist_temp, K=K, random_seed=new_random_seed)
        agent = agent_class(K=K, delta=delta, xi=xi)

        # run the experiment
        round_count = 0
        while not agent.stop:
            arm = agent.action()
            reward = env.response(arm)
            output_arm = agent.observe(reward)
            round_count += 1
            if output_arm is not None:
                break
            if round_count >= stop_benchmark:
                output_arm = -1
                break

        stop_time_[exp_id] = round_count
        if round_count >= stop_benchmark:
            exceed_stop_[exp_id] = 1  # default 0
        if output_arm not in answer_set:
            correctness_[exp_id] = 0

    mean_stop_time = np.mean(stop_time_)
    std_stop_time = np.std(stop_time_) / np.sqrt(n_exp)
    mean_success = np.mean(correctness_)
    count_exceed_stop = np.sum(exceed_stop_)
    count_success = np.sum(correctness_)

    return (
        mean_stop_time,
        std_stop_time,
        mean_success,
        stop_time_,
        correctness_,
        count_exceed_stop,
        count_success,
    )


def Experiment_SEE(
    rlist: np.ndarray,
    delta: float,
    K: int,
    xi: float,
    env_class: Callable,
    agent_class: Callable,
    oracle_class: Callable,
    random_seed_for_start: int = 0,
    disable_tqdm: bool = False,
    shuffle: bool = True,
    stop_benchmark: int = 10**8,
    n_exp: int = 1000,
    delta_k_: Callable = lambda x: 1 / 3 * (5**5) / 5 ** (5**x),
    beta_k_: Callable = lambda x: 4 ** (5**x),
    alpha_k_: Callable = lambda x: 1 / 4 ** (5**x),
):
    """Experiment Oracle

    Args:
        rlist (np.ndarray): reward vector of each arm
        delta (float): confidence level
        K (int): number of arms
        xi (float): the threshold
        env_class (Callable): the environment type
        agent_class (Callable): the 1-identification algorithm
        random_seed_for_start (int, optional): random seed for the experiment oracle. Defaults to 0.
        disable_tqdm (bool, optional): whether output tqdm info. Defaults to False.
        shuffle (bool, optional): whether shuffle the arm. Defaults to True.
        stop_benchmark (int, optional): the maximum length of an experiment. Defaults to 10**8.
        n_exp (int, optional): number of independent experiments. Defaults to 1000.
    """
    assert len(rlist) == K, "number of resources doesn't match"

    np.random.seed(random_seed_for_start)
    random_seed = np.random.randint(low=n_exp, high=n_exp * 100)
    stop_time_ = np.zeros(n_exp)
    correctness_ = np.ones(n_exp)
    exceed_stop_ = np.zeros(n_exp)
    for exp_id in tqdm(range(n_exp), disable=disable_tqdm):
        # shuffle the arm order
        rlist_temp = rlist.copy()
        new_random_seed = random_seed * exp_id
        if shuffle:
            np.random.seed(new_random_seed)
            np.random.shuffle(rlist_temp)
        answer_set = list(np.where(rlist_temp > xi)[0] + 1)
        if len(answer_set) == 0:
            answer_set = ["No Arms Above xi"]

        # define the environment and the agent
        env = env_class(rlist=rlist_temp, K=K, random_seed=new_random_seed)
        agent = agent_class(
            K,
            delta_k_,
            alpha_k_,
            beta_k_,
            oracle_class,
            delta,
            xi,
            C=1.01,
            delta_fraction_C=3.0,
        )

        # run the experiment
        round_count = 0
        while not agent.stop:
            arm = agent.action()
            reward = env.response(arm)
            output_arm = agent.observe(reward)
            round_count += 1
            if output_arm is not None:
                break
            if round_count >= stop_benchmark:
                output_arm = -1
                break

        stop_time_[exp_id] = round_count
        if round_count >= stop_benchmark:
            exceed_stop_[exp_id] = 1  # default 0
        if output_arm not in answer_set:
            correctness_[exp_id] = 0  # default 1

    mean_stop_time = np.mean(stop_time_)
    std_stop_time = np.std(stop_time_) / np.sqrt(n_exp)
    mean_success = np.mean(correctness_)
    count_exceed_stop = np.sum(exceed_stop_)
    count_success = np.sum(correctness_)

    return (
        mean_stop_time,
        std_stop_time,
        mean_success,
        stop_time_,
        correctness_,
        count_exceed_stop,
        count_success,
    )


def Experiment_SEE_recycle(
    rlist: np.ndarray,
    delta: float,
    K: int,
    xi: float,
    env_class: Callable,
    agent_class: Callable,
    random_seed_for_start: int = 0,
    disable_tqdm: bool = False,
    shuffle: bool = True,
    stop_benchmark: int = 10**8,
    n_exp: int = 1000,
    delta_k_: Callable = lambda x: 1 / 3**x,
    beta_k_: Callable = lambda x: 2**x,
    alpha_k_: Callable = lambda x: 5**x,
    sigma_2: float = 1.0,
):
    """Experiment Oracle

    Args:
        rlist (np.ndarray): reward vector of each arm
        delta (float): confidence level
        K (int): number of arms
        xi (float): the threshold
        env_class (Callable): the environment type
        agent_class (Callable): the 1-identification algorithm
        random_seed_for_start (int, optional): random seed for the experiment oracle. Defaults to 0.
        disable_tqdm (bool, optional): whether output tqdm info. Defaults to False.
        shuffle (bool, optional): whether shuffle the arm. Defaults to True.
        stop_benchmark (int, optional): the maximum length of an experiment. Defaults to 10**8.
        n_exp (int, optional): number of independent experiments. Defaults to 1000.
    """
    assert len(rlist) == K, "number of resources doesn't match"

    np.random.seed(random_seed_for_start)
    random_seed = np.random.randint(low=n_exp, high=n_exp * 100)
    stop_time_ = np.zeros(n_exp)
    correctness_ = np.ones(n_exp)
    exceed_stop_ = np.zeros(n_exp)
    for exp_id in tqdm(range(n_exp), disable=disable_tqdm):
        # shuffle the arm order
        rlist_temp = rlist.copy()
        new_random_seed = random_seed * exp_id
        if shuffle:
            np.random.seed(new_random_seed)
            np.random.shuffle(rlist_temp)
        answer_set = list(np.where(rlist_temp > xi)[0] + 1)
        if len(answer_set) == 0:
            answer_set = ["No Arms Above xi"]

        # define the environment and the agent
        env = env_class(rlist=rlist_temp, K=K, random_seed=new_random_seed)
        agent = agent_class(
            K,
            delta_k_,
            alpha_k_,
            beta_k_,
            delta,
            xi,
            C=1.01,
            delta_fraction_C=3.0,
            sigma_2=sigma_2,
        )

        # run the experiment
        round_count = 0
        while not agent.stop:
            arm = agent.action()
            reward = env.response(arm)
            output_arm = agent.observe(reward)
            round_count += 1
            if output_arm is not None:
                break
            if round_count >= stop_benchmark:
                output_arm = -1
                break

        stop_time_[exp_id] = round_count
        if round_count >= stop_benchmark:
            exceed_stop_[exp_id] = 1  # default 0
        if output_arm not in answer_set:
            correctness_[exp_id] = 0  # default 1

    mean_stop_time = np.mean(stop_time_)
    std_stop_time = np.std(stop_time_) / np.sqrt(n_exp)
    mean_success = np.mean(correctness_)
    count_exceed_stop = np.sum(exceed_stop_)
    count_success = np.sum(correctness_)

    return (
        mean_stop_time,
        std_stop_time,
        mean_success,
        stop_time_,
        correctness_,
        count_exceed_stop,
        count_success,
    )


def Experiment_reproduce_Kano(
    rlist: np.ndarray,
    delta: float,
    K: int,
    xi: float,
    env_class: Callable,
    agent_class: Callable,
    random_seed_for_start: int = 0,
    disable_tqdm: bool = False,
    shuffle: bool = True,
    stop_benchmark: int = 10**8,
    n_exp: int = 1000,
    sigma_2: float = 1.0,
):
    """Experiment Oracle

    Args:
        rlist (np.ndarray): reward vector of each arm
        delta (float): confidence level
        K (int): number of arms
        xi (float): the threshold
        env_class (Callable): the environment type
        agent_class (Callable): the 1-identification algorithm
        random_seed_for_start (int, optional): random seed for the experiment oracle. Defaults to 0.
        disable_tqdm (bool, optional): whether output tqdm info. Defaults to False.
        shuffle (bool, optional): whether shuffle the arm. Defaults to True.
        stop_benchmark (int, optional): the maximum length of an experiment. Defaults to 10**8.
        n_exp (int, optional): number of independent experiments. Defaults to 1000.
        sigma_2 (float, optional): subgaussian proxy
    """
    assert len(rlist) == K, "number of resources doesn't match"

    np.random.seed(random_seed_for_start)
    random_seed = np.random.randint(low=n_exp, high=n_exp * 100)
    stop_time_ = np.zeros(n_exp)
    correctness_ = np.ones(n_exp)
    exceed_stop_ = np.zeros(n_exp)
    for exp_id in tqdm(range(n_exp), disable=disable_tqdm):
        # shuffle the arm order
        rlist_temp = rlist.copy()
        new_random_seed = random_seed * exp_id
        if shuffle:
            np.random.seed(new_random_seed)
            np.random.shuffle(rlist_temp)
        answer_set = list(np.where(rlist_temp > xi)[0] + 1)
        if len(answer_set) == 0:
            answer_set = ["No Arms Above xi"]

        # define the environment and the agent
        env = env_class(rlist=rlist_temp, K=K, random_seed=new_random_seed)
        agent = agent_class(K=K, delta=delta, xi=xi, sigma_2=sigma_2)

        # run the experiment
        round_count = 0
        while not agent.stop:
            arm = agent.action()
            reward = env.response(arm)
            output_arm = agent.observe(reward)
            round_count += 1
            if output_arm is not None:
                break
            if round_count >= stop_benchmark:
                output_arm = -1
                break

        stop_time_[exp_id] = round_count
        if round_count >= stop_benchmark:
            exceed_stop_[exp_id] = 1  # default 0
        if output_arm not in answer_set:
            correctness_[exp_id] = 0

    mean_stop_time = np.mean(stop_time_)
    std_stop_time = np.std(stop_time_) / np.sqrt(n_exp)
    mean_success = np.mean(correctness_)
    count_exceed_stop = np.sum(exceed_stop_)
    count_success = np.sum(correctness_)

    return (
        mean_stop_time,
        std_stop_time,
        mean_success,
        stop_time_,
        correctness_,
        count_exceed_stop,
        count_success,
    )


# %% unit test experiment oracle
# from env import Environment_Gaussian
# from agent_SEE import SEE, AdaptedBothUCB
# from agent_Kano import HDoC_Kano

# rlist = np.array([0.5, 0.5, 0.5])
# delta = 0.01
# K = 3
# xi = 0.5
# env_class = Environment_Gaussian
# # agent_class = HDoC_Kano
# agent_class = SEE
# Oracle_Class = AdaptedBothUCB
# random_seed_for_start = 0
# disable_tqdm = False
# shuffle = True
# stop_benchmark = 100
# n_exp = 10

# (
#     _,
#     _,
#     _,
#     _,
#     _,
#     _,
#     _,
# ) = Experiment_SEE(
#     rlist=rlist,
#     delta=delta,
#     K=K,
#     xi=xi,
#     env_class=env_class,
#     agent_class=agent_class,
#     oracle_class=Oracle_Class,
#     random_seed_for_start=random_seed_for_start,
#     disable_tqdm=disable_tqdm,
#     shuffle=shuffle,
#     stop_benchmark=stop_benchmark,
#     n_exp=n_exp,
# )

# %% unit test for Experiment_reproduce_Kano
# from utils_Kano import RealLife
# from agent_APGAI import APGAI

# agent_class = APGAI
# delta = 0.001
# K, rlist, xi, sigma_2, env_class = RealLife()
# n_exp = 1000
# mean_stop_time, std_stop_time, mean_success, _, _, count_exceed_stop, count_success = Experiment_reproduce_Kano(
#     rlist=rlist,
#     delta=delta,
#     K=K,
#     xi=xi,
#     env_class=env_class,
#     agent_class=agent_class,
#     random_seed_for_start=0,
#     n_exp=n_exp,
#     disable_tqdm=False,
#     sigma_2=sigma_2,
# )
# print(
#     f"RealLife, {agent_class.__name__}, {env_class.__name__}, {K},{delta}, done, stop {mean_stop_time}, std {std_stop_time} success {mean_success}"
# )
