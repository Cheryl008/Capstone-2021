import logging
import numpy as np
from abc import abstractmethod, ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence, Dict, Mapping, Optional, List

from .contracts import Asset, Currency, Stock, Option, OptionStyle, OptionFlavor, Trade
from .options_math import black_scholes, andersen
from .params import GBMParams
from .tcost import TransactionCostModel


@dataclass
class MarketDataState:
    asset_prices: Mapping[Asset, float]
    boundary_prices: Mapping[Option, float]

    @property
    def options_at_exercise_boundary(self) -> List[Option]:
        options_exercising_early = []

        for option, early_exercise_price in self.boundary_prices.items():
            assert option.style == OptionStyle.AMERICAN
            stock_price = self.asset_prices[option.underlying]
            should_exercise_early: bool = (
                (option.flavor == OptionFlavor.PUT and stock_price <= early_exercise_price) or
                (option.flavor == OptionFlavor.CALL and stock_price >= early_exercise_price)
            )
            if should_exercise_early:
                options_exercising_early.append(option)

        return options_exercising_early


class MarketDataSource(ABC):
    # set this in ALL subclasses
    universe: Sequence[Asset]

    @abstractmethod
    def get_naive_prices(self, reset: bool = False) -> MarketDataState:
        '''
        This method advances the state of market data by one time step and
        returns a MarketDataState instance, containing two member variables:

            1) asset_prices: a dict mapping each item of the universe to the naive price
               (i.e., NOT adjusted for transaction costs).

            2) boundary_prices: a dict mapping each American option in the
               universe to its boundary price, i.e., the threshold price at which
               it is optimal to exercise the option at the current time if the
               underlying's price is greater (for a call option), or lower (for a
               put option).

        It is the job of a PricingSource object to call this method, and then
        adjust the asset_prices member variable of the returned MarketDataState
        object according to a transaction costs model and pass along a new
        MarketDataState object containing the adjusted prices to a user.

        The PricingSource object should NOT modify the boundary_prices member
        variable of the MarketDataState object returned by
        MarketDataSource.get_naive_prices(); it should just pass that along
        directly to the client, without any adjustments.

        The `reset` parameter indicates whether to reset the MarketDataSource
        instance to an initial state (e.g. the start of a new trading day).
        '''
        pass


class SingleStockGBMMarketDataSource(MarketDataSource):
    @staticmethod
    def validate_universe(universe: Sequence[Asset]):
        stocks = [asset for asset in universe if isinstance(asset, Stock)]
        if len(stocks) != 1:
            raise ValueError(
                "Class SingleStockGBMMarketDataSource is designed to work with exactly one underlying risky asset (stock)"
            )
        stock, = stocks

        cash_assets = [asset for asset in universe if isinstance(asset, Currency)]
        if len(cash_assets) != 1:
            raise ValueError(
                """
                Class SingleStockGBMMarketDataSource is designed to work with exactly one Currency asset.
                In future, multiple currencies will be supported.
                """
            )

        options = [asset for asset in universe if isinstance(asset, Option)]
        assert all(option.underlying == stock for option in options), f"All options in universe must have underlying equal to {stock}"

    def __init__(self, universe: Sequence[Asset], gbm_params: GBMParams, data_reuse_num_episodes: Optional[int] = None):
        SingleStockGBMMarketDataSource.validate_universe(universe)
        self.universe = list(universe)
        self.gbm_params = gbm_params
        self.data_reuse_num_episodes = data_reuse_num_episodes

        # preceding call to validate_universe guarantees exactly 1 Stock and 1 Currency,
        # so next 2 lines are safe
        self._stock, = [asset for asset in self.universe if isinstance(asset, Stock)]
        self._cash, = [asset for asset in self.universe if isinstance(asset, Currency)]

        self._episode_counter: int = -1
        self._episode_cache: List[MarketDataState] = []
        self._episode_cache_idx: Optional[int] = None
        self._market_data_state: MarketDataState

    @property
    def _should_read_from_cache(self):
        return (self.data_reuse_num_episodes is not None) and (self._episode_counter % self.data_reuse_num_episodes != 0)

    @property
    def _should_write_to_cache(self):
        return (self.data_reuse_num_episodes is not None) and (self._episode_counter % self.data_reuse_num_episodes == 0)

    def _reset(self):
        self._episode_counter += 1
        self._curr_time = 0.

        if self._should_read_from_cache:
            self._episode_cache_idx = 0
            self._market_data_state = self._episode_cache[self._episode_cache_idx]
        else:
            # TODO: probably use a different way of setting the initial price ...
            init_stock_price = np.exp(np.log(100)+0.01*np.random.randn())
            init_cash_price = 1.
            self._propagate_prices(init_stock_price, init_cash_price)
            if self._should_write_to_cache:
                self._episode_cache = [self._market_data_state]
                self._episode_cache_idx = None

    def _propagate_prices(self, stock_price: float, cash_price: float):
        asset_prices: Dict[Asset, float] = {self._stock: stock_price, self._cash: cash_price}
        boundary_prices: Dict[Option, float] = {}

        for asset in self.universe:
            if isinstance(asset, Stock):
                assert asset == self._stock
                # we have already set stock price in asset_prices above; no need to do anything here
            elif isinstance(asset, Currency):
                assert asset == self._cash
                # we have already set cash price in asset_prices above; no need to do anything here
            elif isinstance(asset, Option):
                if asset.style == OptionStyle.EUROPEAN:
                    price = black_scholes(
                        tau=asset.expiry_time-self._curr_time,
                        s=stock_price,
                        K=asset.strike,
                        sigma=self.gbm_params.sigma,  # XXX is it correct to pass in an annualized vol?
                        flavor=asset.flavor,
                        r=self.gbm_params.risk_free_rate
                    )
                    asset_prices[asset] = price
                elif asset.style == OptionStyle.AMERICAN:
                    price, boundary_price = andersen(
                        s=stock_price,
                        K=asset.strike,
                        T=asset.expiry_time,
                        t=self._curr_time,
                        sigma=self.gbm_params.sigma,  # XXX is it correct to pass in an annualized vol?
                        flavor=asset.flavor,
                        r=self.gbm_params.risk_free_rate,
                        return_early_exercise_boundary_price=True
                    )
                    asset_prices[asset] = price
                    boundary_prices[asset] = boundary_price
                else:
                    raise ValueError(f"Invalid value {asset.style} for `style` parameter of {asset}")
            else:
                raise TypeError(f"Unable to price asset {asset} in universe")

        self._market_data_state = MarketDataState(
            asset_prices=asset_prices,
            boundary_prices=boundary_prices
        )

    def _update_prices(self):
        self._curr_time += (1/252)

        if self._should_read_from_cache:
            self._episode_cache_idx += 1
            self._market_data_state = self._episode_cache[self._episode_cache_idx]
        else:
            curr_cash_price = self._market_data_state.asset_prices[self._cash]
            new_cash_price = curr_cash_price #*(1+(self.gbm_params.risk_free_rate/252))

            # XXX is it correct to convert mu, sigma to daily values like this?
            curr_stock_price = self._market_data_state.asset_prices[self._stock]
            daily_mu, daily_sigma = (self.gbm_params.mu / 252), (self.gbm_params.sigma / np.sqrt(252))
            daily_log_return = (daily_mu-0.5*daily_sigma**2)+daily_sigma*np.random.randn()
            new_stock_price = curr_stock_price*np.exp(daily_log_return)

            self._propagate_prices(new_stock_price, new_cash_price)
            if self._should_write_to_cache:
                self._episode_cache.append(self._market_data_state)

    def get_naive_prices(self, reset: bool = False) -> MarketDataState:
        if reset:
            self._reset()
        else:
            self._update_prices()
        return deepcopy(self._market_data_state)


class PricingSource:
    def __init__(self, mkt_data_source: MarketDataSource, tcost_model: TransactionCostModel):
        self.mkt_data_source = mkt_data_source
        self.tcost_model = tcost_model

    @property
    def universe(self):
        return self.mkt_data_source.universe

    def update(self, trade: Trade):
        self.tcost_model.update(trade)

    def get_prices(self, reset: bool = False) -> MarketDataState:
        naive_market_data_state: MarketDataState = self.mkt_data_source.get_naive_prices(reset=reset)
        naive_prices: Mapping[Asset, float] = naive_market_data_state.asset_prices
        boundary_prices: Mapping[Option, float] = naive_market_data_state.boundary_prices

        price_adjustments = self.tcost_model.get_price_adjustments(reset=reset)

        for asset in naive_prices:
            if asset not in price_adjustments:
                # TODO maybe we should do more than just warn?
                logging.warn(f"tcost_model returned no price adjustment for asset {asset}. Assuming 0")

        for asset in price_adjustments:
            assert (asset in naive_prices), f"tcost_model returned price adjustment for asset {asset} but mkt_data_source returned no naive price!"

        adjusted_prices = {
            asset: (naive_price + price_adjustments.get(asset, 0))
            for asset, naive_price in naive_prices.items()
        }

        return MarketDataState(
            asset_prices=adjusted_prices,
            boundary_prices=boundary_prices
        )
