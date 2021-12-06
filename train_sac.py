from __future__ import annotations

import itertools
import logging
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
import numpy as np
import pandas as pd
from scipy.stats import norm
import pickle
import sys
sys.path = [pp for pp in sys.path if not '/home/wf541/.local' in pp]
from copy import deepcopy
# from stable_baselines3.ddpg import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC
from torch import nn
import time
from typing import List
from omegaconf import OmegaConf, DictConfig

from golds.callbacks import EvaluationFunctionCallBack, LoggerCallback, ActionFunctionCallback
from golds.contracts import Currency, Stock, Option, OptionFlavor, OptionStyle, Holdings, Valuation
from golds.env import AmericanOptionEnv
from golds.mkt_data import PricingSource, SingleStockGBMMarketDataSource
from golds.params import GBMParams
from golds.reward_functions import NaiveHedgingRewardFunction, RewardFunction
from golds.tcost import NaiveTransactionCostModel
from golds.visualize import make_post_training_plots
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise as OUnoise, NormalActionNoise


def print_usage():
    print("Usage: python3 train_sac.py <path_to_config_yaml>")


def get_training_environment(cfg: DictConfig, save_dir: str):
    aapl = Stock(ticker="AAPL", is_tradable=True)
    # TODO we can also make the option parameters part of the config
    warrant = Option(
        strike=100,
        expiry_time=1.,
        underlying=aapl,
        flavor=OptionFlavor.CALL,
        style=OptionStyle.EUROPEAN,
        is_tradable=False
    )
    cash = Currency(code="USD", is_tradable=False)

    initial_holdings: Holdings = {
        aapl: cfg['init_stock_holdings'],
        warrant: cfg['init_option_holdings'],
        cash: cfg['init_wealth'],
    }

    universe = list(initial_holdings.keys())

    gbm_params = GBMParams(mu=cfg['gbm_mu'], sigma=cfg['gbm_sigma'], risk_free_rate=cfg['gbm_r'])

    mkt_data_source = SingleStockGBMMarketDataSource(universe, gbm_params, data_reuse_num_episodes=cfg['data_reuse_num_episodes'])
    tcost_model = NaiveTransactionCostModel(universe)
    pricing_source = PricingSource(mkt_data_source, tcost_model)

    reward_clip_range = (cfg['reward_clip_min'], cfg['reward_clip_max'])
    reward_records_save_path = os.path.join(save_dir, "reward_history.h5")

    #valid_actions = list(range(cfg['action_min'], cfg['action_max']+1))
    valid_actions = (cfg['action_min'], cfg['action_max'])

    reward_function: RewardFunction = NaiveHedgingRewardFunction(
        kappa=cfg['reward_kappa'],
        initial_holdings=initial_holdings,
        reward_clip_range=reward_clip_range,
        reward_records_save_path=reward_records_save_path
    )

    return AmericanOptionEnv(
        episode_length=50,
        pricing_source=pricing_source,
        reward_function=reward_function,
        actions_config=valid_actions
    )

def get_evaluation_paths(cfg: DictConfig):
    episode_length=cfg['episode_length']
    nepisodes = cfg['num_out_of_sample_path']
    mu=cfg['gbm_mu'] / 252
    sigma=cfg['gbm_sigma'] / np.sqrt(252)
    r=cfg['gbm_r'] / 252
    np.random.seed(42)
    randns = np.random.randn(nepisodes * episode_length)
    res = pd.DataFrame.from_dict({'episode_idx': np.repeat(np.arange(nepisodes), episode_length), 'randn': randns})
    res['time_to_maturity'] = np.tile(1 + np.arange(episode_length), nepisodes) / 252
    res['cum_randn'] = res.groupby('episode_idx')['randn'].cumsum()
    res['price'] = res['cum_randn'] * sigma + (mu - sigma ** 2 / 2 * res['time_to_maturity'] * 252)
    res['price'] = 100 * np.exp(res['price'])
    res['all_1'] = 1.
    res['normalized_time'] = np.tile(1 + np.arange(episode_length), nepisodes) / episode_length
    res['delta'] = norm.cdf((np.log(res['price'] / 100) + (r * 252 + 0.5 * 252 * sigma ** 2) * (1 - res['time_to_maturity'])) /\
        np.sqrt(252) / sigma / np.sqrt((1 - res['time_to_maturity'])))
    res['call_price'] = res['price'] * res['delta'] - 100 * np.exp(-r * 252 * (1 - res['time_to_maturity'])) * norm.cdf((np.log(res['price'] / 100)\
         + (r * 252 - 0.5 * 252 * sigma ** 2) * (1 - res['time_to_maturity'])) / np.sqrt(252) / sigma / np.sqrt((1 - res['time_to_maturity'])))
    return res


def get_observation_grid(env: AmericanOptionEnv) -> List[Valuation]:
    universe = env.pricing_source.universe

    ASSET_PRICE_STEP_SIZE = 0.10
    STOCK_PRICE_MIN = 0.10
    STOCK_PRICE_MAX = 300.00
    STOCK_N_STEPS = int(1+(STOCK_PRICE_MAX-STOCK_PRICE_MIN)/ASSET_PRICE_STEP_SIZE)

    price_grids = []
    for asset in universe:
        if isinstance(asset, Stock):
            price_grids.append(np.linspace(STOCK_PRICE_MIN, STOCK_PRICE_MAX, num=STOCK_N_STEPS))
        elif isinstance(asset, Option):
            option_price_min = 0.
            option_price_max = STOCK_PRICE_MAX - asset.strike
            option_price_num_steps = int(1+(option_price_max-option_price_min)/ASSET_PRICE_STEP_SIZE)
            price_grids.append(np.linspace(option_price_min, option_price_max, num=option_price_num_steps))
        else:
            assert isinstance(asset, Currency)
            price_grids.append(np.array([1.]))

    return [dict(zip(universe, prices)) for prices in itertools.product(*price_grids)]


def main():
    if len(sys.argv) != 2:
        print_usage()
        return 1

    save_dir = os.getcwd()

    cfg = OmegaConf.load(sys.argv[1])

    env = get_training_environment(cfg, save_dir)
    env_copy: AmericanOptionEnv = deepcopy(env)
    # NOTE we make env_copy and call check_env on it, because check_env calls reset() which would mess up the original env
    check_env(env_copy)

    # TODO experiment with gamma, gae_lambda, ent_coef, vf_coef, max_grad_norm (kwargs to PPO.__init__)
    # TODO experiment with batch size (how to do this?)
    # TODO Lerrel says entropy related to exploration -- increase ent_coef if agent is not exploring enough
    # TODO experiment with different number of hidden nodes per layer in "net_arch" (64? 128? more?)
    # TODO reward clipping (*)
    # TODO use t-costs to ensure that the agent does not over-trade
    # TODO check that average reward converges
    # TODO reduce GBM variance such that the entire 50-period episode has vol equivalent to 10 trading days (*)
    # TODO maybe exercise at time of expiry for Euro options (or American without early exercise) and let agent get final reward
    # TODO try continuous action space
    # TODO try transaction costs (this is easily implemented in the RewardFunction.evaluate_reward method)

    logging.info("Hyperparameter settings:")
    for k, v in cfg.items():
        logging.info(f"\t{k} = {v}")

    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    policy_kwargs = {"activation_fn": nn.ReLU, "net_arch": [32]*5}
    # action_noise = OUnoise(mean=np.zeros(1, ), sigma=cfg['OUstd'] * np.ones(1, ), theta=cfg['OUtheta'], dt=cfg['OUdt'])
    SAC_HYPERPARAM_KEYS = ('learning_rate', 'gamma', 'tau', 'train_freq', 'gradient_steps', 'learning_starts')
    sac_hyperparams_dict = {k: cfg[k] for k in SAC_HYPERPARAM_KEYS}
    # model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model = SAC("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, **sac_hyperparams_dict)
    logger_callback = LoggerCallback(save_path=os.path.join(save_dir, "rl_logs.json"), save_freq=10000)
    action_fn_observation_grid = get_evaluation_paths(cfg)
    # action_fn_observation_grid: List[Valuation] = get_observation_grid(env)
    # action_fn_callback = ActionFunctionCallback(model, env, action_fn_observation_grid, save_path=os.path.join(save_dir, "action_fn_logs.h5"), save_freq=10_000)
    logging.info("Getting eval paths")
    with open(os.path.join(save_dir, "eval_path.pkl"), "w+b") as f:
        pickle.dump(action_fn_observation_grid, f)

    evaluator_callback = EvaluationFunctionCallBack(model, env, action_fn_observation_grid, cfg, save_path=[os.path.join(save_dir, "action_fn_logs.h5"), os.path.join(save_dir, "policy_result_logs.h5")], save_freq=100)

    # checkpoint_callback = CheckpointCallback(save_freq=20_000, save_path=save_dir, name_prefix='model_checkpoint')
    # N_YEARS_TRAINING = 50_000
    # TOTAL_TRAINING_TIMESTEPS = N_YEARS_TRAINING*TRADING_DAYS_IN_YEAR
    # model.learn(total_timesteps=cfg['total_training_timesteps'], callback=[logger_callback, action_fn_callback])
    model.learn(total_timesteps=cfg['total_training_timesteps'], callback=[logger_callback, evaluator_callback])

    with open(os.path.join(save_dir, "training_env.pkl"), "w+b") as f:
         pickle.dump(env, f)
    model.save(os.path.join(save_dir, "fully_trained_model"))
    time.sleep(300)
    make_post_training_plots(save_dir, cfg)


if __name__ == '__main__':
    main()