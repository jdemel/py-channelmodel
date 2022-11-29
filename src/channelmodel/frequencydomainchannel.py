#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019, 2020 Johannes Demel.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np


class FrequencyDomainChannel(object):
    def __init__(self, time_variant_channel):
        state = time_variant_channel.state()
        self._fft_len = state['subcarriers']

        self._channel = time_variant_channel
        self._freq_taps = self.calculate_freq_domain_taps()
        self._freq_gains = self.calculate_freq_domain_gains()

    def __repr__(self):
        s = type(self).__name__ + '('
        s += repr(self._channel)
        s += repr(self._freq_taps)
        s += repr(self._freq_gains)
        s += 'subcarriers={}'.format(self._fft_len)
        return s + ')'

    def __str__(self):
        s = type(self).__name__ + '('
        s += 'subcarriers={}'.format(self._fft_len)
        s += ', mean_gain={:.2f}'.format(np.mean(self._freq_gains))
        return s + ')'

    def state(self):
        return self._channel.state()

    def step(self, time_delta=1.e-3):
        self._channel.step(time_delta)
        self._freq_taps = self.calculate_freq_domain_taps()
        self._freq_gains = self.calculate_freq_domain_gains()

    def calculate_freq_domain_taps(self):
        return np.fft.fft(self.time_domain_taps(),
                          self._fft_len).astype(np.complex64)

    def calculate_freq_domain_gains(self):
        return self._freq_taps.real ** 2 + self._freq_taps.imag ** 2

    def subcarriers(self):
        return self._fft_len

    def time_domain_length(self):
        return self._channel.channel_length()

    def time_domain_taps(self):
        return self._channel.channel_taps()

    def freq_domain_taps(self):
        return self._freq_taps

    def freq_domain_gains(self):
        return self._freq_gains
