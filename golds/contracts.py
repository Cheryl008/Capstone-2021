from enum import Enum
from dataclasses import dataclass, astuple
from functools import total_ordering
from typing import Mapping

OptionFlavor = Enum("OptionFlavor", "PUT CALL")

OptionStyle = Enum("OptionStyle", "EUROPEAN AMERICAN")


@total_ordering
@dataclass(frozen=True)
class Asset:
    is_tradable: bool

    def __post_init__(self):
        if type(self) is Asset:
            raise NotImplementedError("Asset is a non-instantiable abstract base class")

    def __lt__(self, other):
        if type(self) is not type(other):
            # This ensures that (e.g.) all "Option" instances will sort before
            # all "Stock" instances.
            return (self.__class__.__name__ < other.__class__.__name__)

        return astuple(self) < astuple(other)


@dataclass(frozen=True)
class Currency(Asset):
    code: str


@dataclass(frozen=True)
class Stock(Asset):
    ticker: str
    # TODO: once we support multiple currencies, we should have Stock contain
    # its pricing Currency as a field here.
    dividend: float = 0.


@dataclass(frozen=True)
class Option(Asset):
    strike: float
    expiry_time: float
    underlying: Stock
    flavor: OptionFlavor
    style: OptionStyle


Valuation = Mapping[Asset, float]

Trade = Mapping[Asset, int]

Holdings = Mapping[Asset, float]
