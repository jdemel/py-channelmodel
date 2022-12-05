#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Johannes Demel.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np


def db2lin(snr_db):
    """db2lin

    A helper function to convert from dB to linear.
    The LaTeX formula would be:
    `10^{value / 10}`.
    Keep in mind that we focus on energy/power values.
    """
    return 10. ** (snr_db / 10.)


def lin2db(snr_lin):
    """lin2db

    A helper function to convert linear energy values to their dB domain equivalent.
    `10 log_{10}(value)`.
    """
    return 10. * np.log10(snr_lin)


def ebn0_to_sigma(ebn0, rate=1.):
    """ebn0_to_sigma

    Generally, `E_b / N_0` is used to define the per bit energy in a simulation.
    e.g. different code rates are better comparable if this parameter is chosen and computed correctly.

    The default is to assume no FEC with `rate == 1`.

    The result is the correct `sigma` value to retrieve random samples with this `sigma` value.
    """
    # Assume E_b = 1.0! Assume complex Gaussian noise N0 = xxx!
    # Assume E_b = 1.0
    ebn0_lin = db2lin(ebn0)
    snr_lin = ebn0_lin * rate
    return np.sqrt(1. / snr_lin)
