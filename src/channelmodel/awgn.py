#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019, 2020 Johannes Demel.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np


def get_complex_noise_vector(vec_len, sigma=1., dtype=np.complex64):
    """ Get complex random vector with variance sigma**2

    We use standard deviation.
    It translates to noise power (variance) with sigma ** 2
    :param vec_len: number of complex noise samples
    :param sigma: standard deviation of noise.
    :return: complex random vector
    """
    # we expect a complex value, needs to be split for I and Q
    # dev = np.sqrt(.5) * sigma
    # noise = np.random.normal(0.0, dev, [2, vec_len])
    # return (noise[0] + 1.j * noise[1]).astype(dtype)
    return get_complex_noise_matrix(vec_len, sigma, dtype)


def get_complex_noise_matrix(shape, sigma=1., dtype=np.complex64):
    dev = np.sqrt(.5) * sigma
    noise_real = np.random.normal(0.0, dev, shape)
    noise_imag = np.random.normal(0.0, dev, shape)
    return (noise_real + 1.j * noise_imag).astype(dtype)


class AWGN(object):
    def __init__(self, ebn0_db, effective_rate=1., subcarriers=1):
        self._snr_db = ebn0_db
        self._effective_rate = effective_rate
        self._subcarriers = subcarriers
        snr_lin = 10. ** (ebn0_db / 10.)
        snr_lin *= effective_rate
        self._sigma = 1. / np.sqrt(1. * snr_lin)
        self._sigma /= np.sqrt(subcarriers)
        self._variance = self._sigma ** 2

    def state(self):
        my_state = {
            'snr': self._snr_db,
            'effective_rate': self._effective_rate,
            'subcarriers': self._subcarriers
        }
        return my_state

    def channel_taps(self):
        return None

    def channel_length(self):
        return 1

    def channel_gains(self):
        return None

    def step(self):
        pass

    def sigma(self):
        return self._sigma

    def variance(self):
        return self._variance

    def snr(self):
        return self._snr_db

    def transmit(self, tx_mod):
        return tx_mod + get_complex_noise_matrix(tx_mod.shape, self._sigma, tx_mod.dtype)
