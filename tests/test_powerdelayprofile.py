#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Johannes Demel.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
import unittest

from .helpers import calculate_average_signal_energy

from channelmodel.powerdelayprofile import PowerDelayProfile, FFTPowerDelayProfile


class PowerDelayProfileTests(unittest.TestCase):
    def setUp(self):
        self._rms_delay_spread = 46.8e-9
        self._max_delay_spread = 250.e-9

    def tearDown(self):
        pass

    def test_setup(self):
        self.assertRaises(NotImplementedError, PowerDelayProfile,
                          self._rms_delay_spread, self._max_delay_spread, 100.e6, shape="penguin")
        self.assertRaises(ValueError, PowerDelayProfile,
                          self._rms_delay_spread, self._max_delay_spread + 1., 100.e6, shape="Exp")
        self.assertRaises(ValueError, PowerDelayProfile,
                          self._rms_delay_spread + 1., self._max_delay_spread, 100.e6, shape="Exp")

    def test_AWGN(self):
        for bandwidth in (20.e6, 50.e6, 100.e6):
            pdp = PowerDelayProfile(self._rms_delay_spread,
                                    self._max_delay_spread, bandwidth)

            s = pdp.supports()
            t = pdp.taps().real
            e = np.sum(np.abs(t) ** 2)
            self.assertAlmostEqual(e, 1.0, 5)
            ntaps = int(np.ceil(bandwidth * self._max_delay_spread))
            self.assertEqual(ntaps, t.size)
            self.assertEqual(ntaps, s.size)

    def test_FFT(self):
        for bandwidth in (20.e6, 50.e6, 100.e6):
            for subcarriers in (32, 64, 135):

                pdp = FFTPowerDelayProfile(self._rms_delay_spread,
                                           self._max_delay_spread,
                                           bandwidth, subcarriers)

                s = pdp.supports()
                t = pdp.taps().real

                ntaps = int(np.ceil(bandwidth * self._max_delay_spread))
                self.assertEqual(ntaps, t.size)
                self.assertEqual(ntaps, s.size)

                f_taps = np.fft.fft(pdp.taps(), subcarriers)
                e = calculate_average_signal_energy(f_taps)
                self.assertAlmostEqual(e, 1.0, 5)
