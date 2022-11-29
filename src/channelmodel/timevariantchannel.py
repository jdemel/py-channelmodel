#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019, 2020 Johannes Demel.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np

from .awgn import get_complex_noise_vector


class TransmissionChannel(object):
    def __init__(self, time_variant_channel):
        self._time_variant_channel = time_variant_channel

    def state(self):
        return self._time_variant_channel.state()

    def transmit(self, tx_symbols):
        h = self._time_variant_channel.channel_taps()
        rx = np.convolve(tx_symbols, h, 'full')[0:tx_symbols.size]
        return rx

    def time_domain_length(self):
        return self._time_variant_channel.channel_length()

    def channel_taps(self):
        return self._time_variant_channel.channel_taps()

    def step(self, time_delta=1.e-3):
        self._time_variant_channel.step(time_delta)


class TimeVariantChannel(object):
    def __init__(self, power_delay_profile):
        self._pdp = power_delay_profile
        self._channel_state = get_complex_noise_vector(
            self._pdp.num_taps())
        self._taps = self._channel_state * self._pdp.taps()

    def state(self):
        my_state = self._pdp.state()
        return my_state

    def update_channel_state(self, time_delta=1.e-3):
        self._channel_state = get_complex_noise_vector(self._pdp.num_taps())

    def update_channel_taps(self):
        self._taps = self._channel_state * self._pdp.taps()

    def step(self, time_delta=1.e-3):
        self.update_channel_state(time_delta)
        self.update_channel_taps()

    def channel_taps(self):
        return self._taps

    def channel_length(self):
        return self._pdp.num_taps()

    def __str__(self):
        s = type(self).__name__ + '('
        s += 'channel_length={}'.format(self.channel_length())
        s += ', mean_gain={:.2f}'.format(np.mean(self._freq_gains))
        return s + ')'


class CoherentTimeVariantChannel(TimeVariantChannel):
    def __init__(self, power_delay_profile, coherence):
        super(CoherentTimeVariantChannel, self).__init__(power_delay_profile)
        self._coherence = coherence

    def state(self):
        s = super(CoherentTimeVariantChannel, self).state()
        s.update(self._coherence.state())
        return s

    def current_weight(self, covariance):
        return np.sqrt(covariance)

    def next_weight(self, covariance):
        return np.sqrt(1 - covariance)

    def update_channel_state(self, time_delta=1.e-3):
        cov = self._coherence.coherence_time(time_delta)
        n = get_complex_noise_vector(self._pdp.num_taps())
        self._channel_state *= self.current_weight(cov)
        self._channel_state += self.next_weight(cov) * n
