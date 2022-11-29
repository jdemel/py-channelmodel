#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019, 2020, 2021 Johannes Demel.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np


def calculate_vector_signal_energy(s):
    return np.sum(s.real ** 2 + s.imag ** 2)


class PowerDelayProfile(object):
    def __init__(self, rms_delay_spread, max_delay_spread, bandwidth,
                 scale=1., shape='Exp'):
        if shape != 'Exp':
            raise NotImplementedError("Currently only 'Exp' supported!")
        if rms_delay_spread > 1.e-5:
            raise ValueError('This "{}" value does not make sense'.format(rms_delay_spread))
        if max_delay_spread > 1.e-4:
            raise ValueError('This "{}" value does not make sense'.format(max_delay_spread))

        self._shape = shape
        self._rms_delay_spread = rms_delay_spread
        self._max_delay_spread = max_delay_spread
        self._bandwidth = bandwidth
        samp_dur = 1. / bandwidth
        num_taps = int(np.ceil(max_delay_spread / samp_dur))
        self._supports = np.arange(0., max_delay_spread, samp_dur)
        assert self._supports.size == num_taps
        self._scale = scale
        self.initialize_pdp()

    def state(self):
        s = {'rms_delay_spread': self._rms_delay_spread,
             'max_delay_spread': self._max_delay_spread,
             'bandwidth': self._bandwidth,
             'scale': self._scale,
             'shape': self._shape}
        return s

    def samp_ticks(self):
        return self._supports[1] - self._supports[0]

    def num_taps(self):
        return self._supports.size

    def initialize_pdp(self):
        self._pdp = self.exp_dist(self._supports, self._rms_delay_spread)
        self._pdp /= np.sqrt(calculate_vector_signal_energy(self._pdp))
        pdp_energy = calculate_vector_signal_energy(self._pdp)
        assert np.abs(pdp_energy - 1.) < 1e-13
        self._pdp *= self._scale
        pdp_energy = calculate_vector_signal_energy(self._pdp)
        assert np.abs(np.sqrt(pdp_energy) - self._scale) < 1e-13
        assert self._pdp.size == self.num_taps()
        self._pdp = self._pdp.astype(np.complex64)

    def exp_dist(self, supports, beta):
        return np.exp(-1. * supports / beta) / beta

    def taps(self):
        return self._pdp

    def supports(self):
        return self._supports


class FFTPowerDelayProfile(PowerDelayProfile):
    def __init__(self, rms_delay_spread, max_delay_spread, bandwidth,
                 subcarriers, shape='Exp'):
        scale = 1.
        super(FFTPowerDelayProfile, self).__init__(rms_delay_spread,
                                                   max_delay_spread, bandwidth,
                                                   scale=scale, shape=shape)

        f_taps = np.fft.fft(self.taps(), subcarriers)
        assert f_taps.size == subcarriers
        e = calculate_vector_signal_energy(f_taps)

        self._scale = 1. / np.sqrt(1. * e / subcarriers)

        self.initialize_pdp()
        freq_taps = np.fft.fft(self.taps(), subcarriers)
        assert np.mean(np.abs(freq_taps) ** 2) - 1. < 1.e-6
        self._subcarriers = subcarriers

    def state(self):
        s = super(FFTPowerDelayProfile, self).state()
        s['subcarriers'] = self._subcarriers
        return s

    def subcarriers(self):
        return self._subcarriers


if __name__ == '__main__':
    print('boo')
    pdp = PowerDelayProfile(46e-9, 250e-9, 100e6)
    print(pdp.num_taps())
    print(pdp.taps())
    powers = 10. * np.log10(pdp.taps().real ** 2)
    powers -= np.amax(powers)
    print(powers)
