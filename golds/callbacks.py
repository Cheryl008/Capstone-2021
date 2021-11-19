import logging; logging.basicConfig(level=logging.INFO)

import gc
import json
import numpy as np
import pandas as pd

from typing import Any, List

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import logger

from .contracts import Valuation
from .env import AmericanOptionEnv, Continuity


def _json_encode_numpy(val: Any) -> Any:
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()

    raise TypeError(f"Cannot JSON serialize object {val} of type {type(val)}")


class LoggerCallback(BaseCallback):
    def __init__(self, save_path: str, save_freq: int, verbose: int = 0):
        super(LoggerCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            with open(self.save_path, 'a') as f:
                json.dump(logger.get_log_dict(), f, default=_json_encode_numpy)
                f.write('\n')

        return True


class ActionFunctionCallback(BaseCallback):
    def __init__(self, agent: PPO, env: AmericanOptionEnv, observation_grid: List[Valuation], save_path: str, save_freq: int, verbose: int = 0):
        super(ActionFunctionCallback, self).__init__()
        self.agent = agent
        self.universe = env.pricing_source.universe
        self.save_path = save_path
        self.save_freq = save_freq

        # Save observation_grid to file
        start = pd.Timestamp.now()

        observation_grid_df: pd.DataFrame = pd.DataFrame.from_records(observation_grid)
        observation_grid_df.to_hdf(save_path, mode='w', key='observation_grid', format='table')

        del observation_grid_df
        gc.collect()

        logging.info(f"Took {pd.Timestamp.now()-start} to construct and persist observation_grid_df to HDF5")

        # Store observation_grid as list of arrays, as this will be what is passed into the agent
        self.observation_array_grid = [env.observation_dict_to_array(obs) for obs in observation_grid]
        assert all(len(obs) == len(self.universe) for obs in self.observation_array_grid)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            start = pd.Timestamp.now()

            action_fn_records: List[dict] = []
            for obs in self.observation_array_grid:
                action_fn_record = {'timestep': self.n_calls}
                for asset, price in zip(self.universe, obs):
                    action_fn_record[asset] = price

                action, _states = self.agent.predict(obs)
                action_fn_record['action'] = json.dumps(_json_encode_numpy(action))

                action_fn_records.append(action_fn_record)

            action_fn_df: pd.DataFrame = pd.DataFrame.from_records(action_fn_records)
            action_fn_df.to_hdf(self.save_path, mode='a', key='action_fn_logs', append=True, format='table')

            del action_fn_df
            gc.collect()

            logging.info(f"ActionFunctionCallback took {pd.Timestamp.now()-start} to evaluate action function on observation grid")

        return True

class EvaluationFunctionCallBack(BaseCallback):
    def __init__(self, agent: PPO, env: AmericanOptionEnv, observation_grid: pd.DataFrame, cfg, save_path: str, save_freq: int, verbose: int = 0):
        super(EvaluationFunctionCallBack, self).__init__()
        self.agent = agent
        self.universe = env.pricing_source.universe
        self.save_path = save_path
        self.save_freq = save_freq
        self.cfg = cfg
        self.env = env
        # if env._action_space_continuity == Continuity.CONTINUOUS:
            # raise NotImplementedError()

        self.obs_input = observation_grid
        self.neposides = self.obs_input['episode_idx'].max() + 1

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            start = pd.Timestamp.now()
            
            self.obs_input = self.obs_input.sort_values(['normalized_time', 'episode_idx'])
            prev_pos = None
            preds = []
            for _t in sorted(self.obs_input['normalized_time'].unique()):
                prev_pos = np.ones(self.neposides) * self.cfg['init_stock_holdings'] if prev_pos is None else prev_pos
                obs = np.hstack([self.obs_input.loc[self.obs_input['normalized_time'] == _t, ['price', 'all_1', 'call_price', 'normalized_time',]].values,
                    prev_pos[:, np.newaxis], np.zeros((self.neposides, 2))])
                if self.env._action_space_continuity == Continuity.DISCRETE:
                    pred = self.agent.predict(obs)[0].T[0] + self.cfg['action_min']  # TODO do not hack this
                elif self.env._action_space_continuity == Continuity.INT:
                    pred = self.agent.predict(obs)[0] + self.cfg['action_min'] # TODO do not hack this
                elif self.env._action_space_continuity == Continuity.CONTINUOUS:
                    pred = self.agent.predict(obs)[0].T[0]
                prev_pos = pred
                preds.append(pred)

            self.obs_input['holdings'] = np.concatenate(preds)
            self.obs_input = self.obs_input.sort_values(['episode_idx', 'normalized_time'])
            self.obs_input = self.calculate_reward_vectorize(self.obs_input, self.cfg)

            # we serialize only holdings and reward, due to the deterministic nature of paths we generated
            action_fn_df: pd.DataFrame = self.obs_input[['holdings', 'reward']]
            action_fn_df.to_hdf(self.save_path, mode='a', key='action_fn_logs', append=True, format='table')

            logging.info(f"ActionFunctionCallback took {pd.Timestamp.now()-start} to evaluate action function on observation grid")

        return True

    @staticmethod
    def calculate_reward_vectorize(res, cfg):
        res['delta_wealth'] = res['holdings'].shift() * res['price'] + 100 * res['call_price'] - (res['holdings'] * res['price'] + 100 * res['call_price']).shift()
        res.loc[res['normalized_time']==res['normalized_time'].min(), 'delta_wealth'] = 0.
        res['reward'] = res['delta_wealth'] ** 2 * -0.5 * cfg['reward_kappa']  # this is reward function
        return res

