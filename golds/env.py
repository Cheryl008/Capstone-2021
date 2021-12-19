import numpy as np

from enum import Enum
from operator import xor
from typing import Union, Mapping, Tuple, List, Dict

from gym import Env
from gym import spaces

from .config import logging
from .contracts import Asset, Trade, Valuation
from .mkt_data import MarketDataState, PricingSource
from .reward_functions import RewardFunction

import itertools

ActionsConfig = Union[List[int], Mapping[Asset, List[int]], Tuple[float, float], Mapping[Asset, Tuple[float]], Dict[str, int]]

Continuity = Enum("Continuity", "CONTINUOUS DISCRETE INT")


class AmericanOptionEnv(Env):
    def __init__(
        self,
        episode_length: int,
        pricing_source: PricingSource,
        reward_function: RewardFunction,
        actions_config: ActionsConfig
    ):
        self.episode_length = episode_length
        self.pricing_source = pricing_source
        self.reward_function = reward_function

        self._tradable_universe = [asset for asset in pricing_source.universe if asset.is_tradable]
        self._initialize_action_space(actions_config)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(int(len(pricing_source.universe) * 2 + 1),), dtype=np.float32)
        self._i = 0  # index in current episode

        # TODO: (optionally) set self.reward_space

    def _validate_actions_config_tuple(self, actions_config: Tuple) -> Continuity:
        try:
            lo, hi = actions_config
            lo = float(lo)
            hi = float(hi)
            assert lo < hi
        except (ValueError, AssertionError):
            raise ValueError("If actions_config is a tuple, it must be a tuple of exactly 2 floats specifying a nonempty interval")

        return Continuity.CONTINUOUS

    def _validate_actions_config_list(self, actions_config: List) -> Continuity:
        try:
            actions_config = list(map(int, actions_config))
        except ValueError:
            raise ValueError("If actions_config is a list, it must be a list of ints")

        return Continuity.DISCRETE

    def _validate_actions_config_dictionary(self, actions_config: Dict) -> Continuity:
        try:
            lo = int(actions_config['lo'])
            hi = int(actions_config['hi'])
            assert lo < hi
        except ValueError:
            raise ValueError("If actions_config is a dictionary, it must be a dictionary with two elements, low and high")

        return Continuity.INT

    def _validate_actions_config_mapping(self, actions_config: Mapping) -> Continuity:
        assert sorted(actions_config.keys()) == sorted(self._tradable_universe), "If action_config is a mapping, its keyset must be equal to the tradable universe"

        values_all_sequence_of_ints: bool = False
        values_all_pairs_of_floats: bool = False

        try:
            all_actions = []
            for value in actions_config.values():
                value_as_sequence_of_ints = list(map(int, value))
                all_actions.extend(value_as_sequence_of_ints)
            values_all_sequence_of_ints = True
        except ValueError:
            pass

        try:
            for value in actions_config.values():
                lo, hi = map(float, value)
                assert lo < hi
            values_all_pairs_of_floats = True
        except (ValueError, AssertionError):
            pass

        if not xor(values_all_sequence_of_ints, values_all_pairs_of_floats):
            raise ValueError(
                """
                If actions_config is a mapping, the values must either all be sequences of ints,
                or all be pairs of floats specifying nonempty intervals
                """
            )

        return (Continuity.DISCRETE if values_all_sequence_of_ints else Continuity.CONTINUOUS)

    def _validate_actions_config(self, actions_config: ActionsConfig) -> Continuity:
        '''
        This method performs various checks to ensure that actions_config is a valid action space configuration,
        and returns a value of the Continuity enum indicating whether the action space is discrete or continuous.
        '''
        if isinstance(actions_config, Tuple):
            return self._validate_actions_config_tuple(tuple(actions_config))
        elif isinstance(actions_config, List):
            return self._validate_actions_config_list(actions_config)
        elif isinstance(actions_config, Dict):
            return self._validate_actions_config_dictionary(actions_config)
        else:
            if not isinstance(actions_config, Mapping):
                raise ValueError("actions_config must be a tuple, a list, or a mapping")
            return self._validate_actions_config_mapping(actions_config)

    def _initialize_action_space(self, actions_config: ActionsConfig):
        self._action_space_continuity = self._validate_actions_config(actions_config)

        if self._action_space_continuity == Continuity.DISCRETE:
            # NOTE: `gym.spaces.MultiDiscrete` takes a sequence of positive
            # integers `a_maxs` and generates a discrete action space which is the
            # product of `([0, ..., a_max-1] for a_max in a_maxs)`
            #
            # Therefore, we cache `actions_config` internally as `self._actions`, a
            # dict mapping each asset in the tradable universe to the list of all
            # possible economic actions (i.e., units to buy/sell). Then, we
            # construct a self.action_space as a MultiDiscrete space of
            # `[len(self._actions[asset]) for asset in self._tradable_universe]`.
            # Now, it is possible to "map back" from an action vector (vector of
            # indices, one for each asset) to the space of actual actions (units
            # bought/sold for each asset). This is done in the
            # AmericanOptionsEnv.action_array_to_dict method below.
            _actions: Dict[Asset, List[int]] = {}

            if isinstance(actions_config, Mapping):
                for asset, possible_trade_sizes in actions_config.items():
                    _actions[asset] = list(map(int, possible_trade_sizes))
            else:
                assert isinstance(actions_config, list)
                possible_trade_sizes: List[int] = list(map(int, actions_config))
                for asset in self._tradable_universe:
                    _actions[asset] = possible_trade_sizes

            action_space: spaces.Space = spaces.MultiDiscrete([len(_actions[asset]) for asset in self._tradable_universe])
        
        elif self._action_space_continuity == Continuity.INT:
            _actions: List[Tuple] = []

            assert isinstance(actions_config, Dict)

            lo = int(actions_config['lo'])
            hi = int(actions_config['hi'])

            possible_trade_sizes: List[int] = list(range(lo, hi+1))
            _actions = list(itertools.product(possible_trade_sizes, repeat=len(self._tradable_universe)))
            count = len(possible_trade_sizes) ** len(self._tradable_universe)

            action_space: spaces.Space = spaces.Discrete(count)

        else:  # self._action_space_continuity == Continuity.CONTINUOUS
            _actions: Dict[Asset, Tuple[float, float]] = {}

            if isinstance(actions_config, tuple):
                lo, hi = map(float, actions_config)
                for asset in self._tradable_universe:
                    _actions[asset] = (lo, hi)
            else:
                assert isinstance(actions_config, Mapping)
                for asset, action in actions_config.items():
                    lo, hi = map(float, action)
                    _actions[asset] = (lo, hi)

            action_space: spaces.Space = spaces.Box(
                low=np.array([_actions[asset][0] for asset in self._tradable_universe]),
                high=np.array([_actions[asset][1] for asset in self._tradable_universe]),
                dtype=np.float32
            )

        self._actions: Union[Dict[Asset, Tuple[float, float]], Dict[Asset, List[int]]] = _actions
        self.action_space: spaces.Space = action_space

    def step(self, action):
        self._i += 1

        logging.info(f"Entering step {self._i} of {self.episode_length} | action = {action}")

        trade: Trade = self.action_array_to_dict(action, round_lots=True)

        logging.info(f"Action: {action} Trade {trade}")

        self.pricing_source.update(trade)
        market_data_state: MarketDataState = self.pricing_source.get_prices()

        # TODO: instead of terminating the episode early when an option is exercised, keep the episode going to the end.
        # However, the option will no longer be part of the portfolio (so in general, the agent's trading behavior will cause variance of wealth to be high.)
        # This will discourage the agent from trading after exercise.
        #
        # We should also inform the agent that exercise has occurred, as an observation. Should also tell the agent the delta (as an observation).
        reward, terminate_early = self.reward_function.evaluate_reward(trade, market_data_state)

        prices: Mapping[Asset, float] = market_data_state.asset_prices
        observation = self.observation_dict_to_array(prices)
        observation = np.append(observation, [self._i / self.episode_length] + [trade.get(asset, 0) for asset in self.pricing_source.universe]).astype(np.float32)

        done = (self._i == self.episode_length) or terminate_early

        logging.info(f"Returning from step {self._i} of {self.episode_length} | observation={observation}, reward={reward}, done={done}")

        return observation, reward, done, {}

    def reset(self):
        initial_market_data_state: MarketDataState = self.pricing_source.get_prices(reset=True)
        assert not initial_market_data_state.options_at_exercise_boundary, "No American options should be optimally-exercised at time 0"
        initial_prices: Mapping[Asset, float] = initial_market_data_state.asset_prices

        observation = self.observation_dict_to_array(initial_prices)
        self.reward_function.reset(initial_prices)
        self._i = 0
        return np.append(observation, [self._i / self.episode_length] + \
            [self.reward_function.initial_holdings[asset] if asset in self._tradable_universe else 0 for asset in self.pricing_source.universe]).astype(np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def observation_array_to_dict(self, observation: np.ndarray) -> Valuation:
        # TODO: temporarily change the shape to be prices, positions and time-to-maturity
        assert observation.shape == (int(len(self.pricing_source.universe) * 2 + 1),)
        return dict(zip(self.pricing_source.universe, observation))

    def observation_dict_to_array(self, observation: Valuation) -> np.ndarray:
        # NOTE: For consistency with the semantics of `self.observation_space`
        # (which is a `spaces.Box` of shape `(len(pricing_source.universe),)`),
        # we need to convert the dict `prices` into a `numpy.ndarray` where the
        # i-th value is the price of asset `self.pricing_source.universe[i]`
        return np.array([observation[asset] for asset in self.pricing_source.universe], dtype=np.float32)

    def action_array_to_dict(self, action: np.ndarray, round_lots: bool = False) -> Trade:
        # assert action.shape == (len(self._tradable_universe),)

        if self._action_space_continuity == Continuity.DISCRETE:
            # NOTE: Remember that, by the semantics of `spaces.MultiDiscrete`,
            # the `action` argument to this function will a numpy.ndarray of
            # shape `(len(self._tradable_universe),)`.  The i-th value in the
            # array will be an index into
            # `self._actions[self._tradable_universe[i]]` (an array
            # representing the economic actions [i.e., amount to buy or sell]
            # that our system expects).  So, we need to "map back" into our
            # internal action space as follows.
            return {
                asset: self._actions[asset][action_idx]
                for asset, action_idx in zip(self._tradable_universe, action)
            }
        elif self._action_space_continuity == Continuity.INT:
            result_dic = {}
            for i in range(len(self._actions[action])):
                asset = self._tradable_universe[i]
                selected_action = self._actions[action][i]
                result_dic[asset] = selected_action
            return result_dic
        else:  # self._action_space_continuity == Continuity.CONTINUOUS
            if round_lots:
                return dict(zip(self._tradable_universe, map(round, action)))
            else:
                return dict(zip(self._tradable_universe, action))

    def action_dict_to_array(self, action: Trade) -> np.ndarray:
        if self._action_space_continuity == Continuity.DISCRETE:
            action_array = np.array([self._actions[asset].index(int(action[asset])) for asset in self._tradable_universe], dtype=np.float32)
        else:  # self._action_space_continuity == Continuity.CONTINUOUS
            action_array = np.array([action[asset] for asset in self._tradable_universe], dtype=np.float32)

        assert self.action_space.contains(action_array)
        return action_array
