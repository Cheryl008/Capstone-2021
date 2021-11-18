from abc import ABC, abstractmethod
from typing import Sequence, Mapping

from .contracts import Asset, Trade


class TransactionCostModel(ABC):
    # set in ALL subclasses
    universe: Sequence[Asset]

    @abstractmethod
    def update(self, action):
        pass

    @abstractmethod
    def get_price_adjustments(self, reset: bool = False) -> Mapping[Asset, float]:
        pass


class NaiveTransactionCostModel(TransactionCostModel):
    def __init__(self, universe: Sequence[Asset]):
        self.universe = list(universe)

    def update(self, trade: Trade):
        pass

    def get_price_adjustments(self, reset: bool = False) -> Mapping[Asset, float]:
        return {asset: 0. for asset in self.universe}

    def reset(self):
        pass


class LinearQuadraticTransactionCostModel(TransactionCostModel):
    # TODO add other parameters as needed ...
    def __init__(self, universe: Sequence[Asset], eta: float):
        self.universe = list(universe)
        self.eta = eta

        self._price_adjustments = {asset: 0. for asset in universe}

    def update(self, trade: Trade):
        # TODO implement this
        for asset in trade:
            assert asset in self.universe
            self._price_adjustments[asset] += 0.1234

    def get_price_adjustments(self, reset: bool = False) -> Mapping[Asset, float]:
        self._reset()
        return dict(self._price_adjustments)

    def _reset(self):
        # XXX this is quite likely wrong if we want to retain permanent impact
        self._price_adjustments = {asset: 0. for asset in self.universe}
