from dataclasses import dataclass


@dataclass(frozen=True)
class GBMParams:
    mu: float
    sigma: float
    risk_free_rate: float  # XXX: strictly speaking this is not a GBM param; maybe have separate EconomicParams struct?
