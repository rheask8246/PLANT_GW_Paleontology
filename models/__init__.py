"""Models for rate prediction, CFM emulation, and amortized population posteriors."""
from .rate_network import RateNet, NormalizedNet, RateNetwork, NormalizedRateNet
from .cfm_emulator import CFMEmulator, EventEncoder, VectorFieldNet, generate_catalog, normalize_obs, denormalize_obs
from .naive_bayes_emulator import NaiveBayesEmulator
from .naive_bayes_emulator import generate_catalog as generate_catalog_nb
from .posterior_network_lite import (
    THETA_DIM,
    LitePosteriorNet,
    PosteriorNet,
    SSPC_THETA_PARAM_COLS,
)
from .posterior_network_full import FullPosteriorNet

__all__ = [
    "RateNet",
    "NormalizedNet",
    "RateNetwork",
    "NormalizedRateNet",
    "CFMEmulator",
    "EventEncoder",
    "VectorFieldNet",
    "generate_catalog",
    "normalize_obs",
    "denormalize_obs",
    "NaiveBayesEmulator",
    "generate_catalog_nb",
    "THETA_DIM",
    "SSPC_THETA_PARAM_COLS",
    "PosteriorNet",
    "LitePosteriorNet",
    "FullPosteriorNet",
]
