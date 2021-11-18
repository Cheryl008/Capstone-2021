import numpy as np
from functools import lru_cache
from scipy.stats import norm

from .config import logging
from .contracts import OptionFlavor
from .py_andersen import AndersenModel


# This code was taken almost verbatim from the pkolm/OptionsHedging repo.
# I have modified it slightly to return put prices as well as call prices.
def black_scholes(tau: float, s: float, K: float, sigma: float, flavor: OptionFlavor, r: float = 0, return_aux_values: bool = False):
    '''
    Computes a set of theoretical quantities associated to a European option in the Black-Scholes model.

    :param tau: time to expiry
    :param S: the underlying's value
    :param K: the strike of the option
    :param sigma: the implied volatility
    :param flavor: the option flavor (must be OptionFlavor.PUT or OptionFlavor.CALL)
    :param r: risk-free interest
    :param return_aux_values: whether or not to return delta, d1, d2 in addition to price
    :return:
        if return_aux_values:
            price, delta, d1, d2
        else:
            price
    '''
    if tau > 0:
        val1 = np.log(s / K)
        val2 = (r + 0.5 * (sigma ** 2)) * tau
        val3 = sigma * np.sqrt(tau)
        d1 = (val1 + val2) / val3
        d2 = d1 - (sigma * np.sqrt(tau))
        delta = norm.cdf(d1)
        call_price = s * delta - K * (np.e ** (-tau * r)) * norm.cdf(d2)
        # TODO should incorporate dividend rate when using put-call parity ...
        B = np.exp(-r*tau)
        put_price = call_price - s + K*B
    else:
        d1 = 0
        d2 = 0
        delta = norm.cdf(d1)
        call_price = np.clip(s - K, 0., np.infty)
        put_price = np.clip(K - s, 0, np.infty)

    price = call_price if flavor == OptionFlavor.CALL else put_price
    delta = delta if flavor == OptionFlavor.CALL else -delta
    if return_aux_values:
        return price, delta, d1, d2
    else:
        return price


def andersen(s: float, K: float, T: float, t: float, sigma: float, flavor: OptionFlavor, r: float = 0., return_early_exercise_boundary_price: bool = False):
    call_or_put_str = ("call" if flavor == OptionFlavor.CALL else "put")
    andersen_model = AndersenModel(
        s, K, T, t, r, 0.0, sigma, call_or_put_str,
        32, 8, 54, 101  # these parameters relate to the Gaussian quadrature
    )
    andersen_model.calculateB()
    # XXX For some reason, non-zero r parameter causes price to be nan here ...
    price = andersen_model.priceAt(s, t)
    logging.debug(f"andersen(T={T}, t={t}, tau={T-t}, s={s}, K={K}, sigma={sigma}, flavor={flavor}, r={r}) = {price}")

    if return_early_exercise_boundary_price:
        tau: float = T - t
        # XXX you need to pass in t, rather than tau, to andersen_model.getBoundaryPrice
        # TODO fix semantics here with Yucheng
        early_exercise_boundary_price: float = andersen_model.getBoundaryPrice(t)
        logging.debug(f"AndersenModel early exercise boundary price = {early_exercise_boundary_price}")
        return price, early_exercise_boundary_price
    else:
        return price