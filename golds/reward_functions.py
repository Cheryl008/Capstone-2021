import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain
from typing import DefaultDict, Tuple, Mapping, List, Optional

from .contracts import Currency, Trade, Valuation, Asset, Holdings, Option, OptionFlavor
from .mkt_data import MarketDataState
from .config import logging

import math


def _value_of_holdings(holdings: Holdings, prices: Valuation, prev_holdings: Holdings = None) -> float:
    if prev_holdings is not None:
        return sum(holdings.get(asset, qty) * prices[asset] for asset, qty in prev_holdings.items())
    else:
        return sum(qty * prices[asset] for asset, qty in holdings.items())


class RewardFunction(ABC):
    @abstractmethod
    def evaluate_reward(self, trade: Trade, market_data_state: MarketDataState) -> Tuple[float, bool]:
        pass

    @abstractmethod
    def reset(self):
        pass


class NaiveHedgingRewardFunction(RewardFunction):
    '''
    This class implements the transaction cost-naive reward function given in
    equation (22) of Du, Jin, Kolm, et al. The full paper may be downloaded
    here: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3677201
    '''
    def __init__(self, kappa: float, initial_holdings: Holdings, reward_clip_range: Optional[Tuple[float, float]] = None, reward_records_save_path: Optional[str] = None):
        self.kappa = kappa
        self.initial_holdings: Holdings = dict(initial_holdings)
        self.reward_clip_range = reward_clip_range
        self.reward_records_save_path = reward_records_save_path

        self._episode_counter: int = -1
        self._reward_records = []
        self._reward_records_df = pd.DataFrame()

        # NOTE: You must call `NaiveRewardFunction.reset` with an initial dict
        # of prices before you can call `NaiveRewardFunction.evaluate_reward`
        # for the first time.

    def reset(self, initial_prices):
        self._episode_counter += 1
        self._idx_in_episode = 0
        self._prev_holdings: Holdings = self.initial_holdings
        self._prev_prices: Valuation = dict(initial_prices)

        SAVE_RECORDS_NUM_EPISODES = 100

        if self._reward_records:
            # TODO it might be best to save the reward records from each episode using a Callback,
            # but how to implement a callback that runs at the end of each episode?
            reward_records_df = pd.DataFrame.from_records(self._reward_records)
            min_itemsize = {}
            for col in ('prev_prices', 'curr_prices', 'trade', 'prev_holdings', 'curr_holdings', 'boundary_prices'):
                reward_records_df[col] = reward_records_df[col].astype(str)
                min_itemsize[col] = (1<<9)
            self._reward_records_df = pd.concat((self._reward_records_df, reward_records_df), ignore_index=True)
            if self._episode_counter % SAVE_RECORDS_NUM_EPISODES == 0:
                self._reward_records_df.to_hdf(
                    self.reward_records_save_path,
                    mode='a', key='df', append=True, format='table',
                    min_itemsize=min_itemsize
                )
                self._reward_records_df = pd.DataFrame()
            self._reward_records = []

    def evaluate_reward(self, trade: Trade, market_data_state: MarketDataState) -> Tuple[float, bool]:
        prices: Valuation = market_data_state.asset_prices
        boundary_prices: Mapping[Option, float] = market_data_state.boundary_prices

        for asset in trade.keys():
            assert asset.is_tradable, f"Invalid attempt to trade non-tradable asset {asset}"

        # Compute revenue due to trade
        trade_value: float = _value_of_holdings(trade, self._prev_prices, self._prev_holdings) - _value_of_holdings(self._prev_holdings, self._prev_prices)
        cash_assets: List[Asset] = [asset for asset in prices.keys() if isinstance(asset, Currency)]
        try:
            cash, = cash_assets
        except ValueError:
            raise NotImplementedError(
                f'''
                Require exactly one Currency asset but found: {cash_assets}.
                Multiple currencies not currently supported.
                '''
            )
        revenue = -(trade_value / prices[cash])

        # Update asset holdings due to trade

        curr_holdings: DefaultDict[Asset, float] = defaultdict(float)
        for asset, qty in chain(self._prev_holdings.items(), trade.items()):
            curr_holdings[asset] = qty

        # Update revenue and asset holdings due to early exercise of American options (if any)

        options_at_exercise_boundary: List[Option] = market_data_state.options_at_exercise_boundary

        # NOTE: this will terminate the episode if ANY of our American options are being exercised early
        terminate_early: bool = bool(options_at_exercise_boundary)

        for option in options_at_exercise_boundary:
            stock_price: float = prices[option.underlying]
            payoff: float = (stock_price - option.strike) if option.flavor == OptionFlavor.CALL else (option.strike - stock_price)
            assert payoff > 0.
            revenue += curr_holdings[option]*payoff
            curr_holdings[option] = 0

        # Finally, update holdings of cash due to revenue from trades and/or option exercise

        curr_holdings[cash] += revenue

        prev_wealth = _value_of_holdings(self._prev_holdings, self._prev_prices)
        curr_wealth = _value_of_holdings(curr_holdings, prices)
        delta_wealth = curr_wealth - prev_wealth
        # the scale of kappa should be proportional to inverse of delta wealth
        reward = - (self.kappa/2)*(delta_wealth ** 2) + delta_wealth

        # (Optional) reward clipping

        if self.reward_clip_range:
            min_reward, max_reward = self.reward_clip_range
            reward = np.clip(reward, min_reward, max_reward)

        record = {
            'episode_num': self._episode_counter,
            'idx_in_episode': self._idx_in_episode,
            'prev_holdings': self._prev_holdings,
            'prev_prices': self._prev_prices,
            'prev_wealth': prev_wealth,
            'trade': trade,
            'revenue': revenue,
            'curr_holdings': curr_holdings,
            'curr_prices': prices,
            'curr_wealth': curr_wealth,
            'delta_wealth': delta_wealth,
            'reward': reward,
            'boundary_prices': boundary_prices,
            'terminate_early': terminate_early
        }
        self._reward_records.append(record)

        self._prev_holdings = dict(curr_holdings)
        self._prev_prices = dict(prices)
        self._idx_in_episode += 1

        return reward, terminate_early

    def persist_history_to_hdf(self, *args, **kwargs):
        pd.DataFrame.from_records(self._reward_records).to_hdf(*args, **kwargs)
