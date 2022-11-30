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

from channelmodel.awgn import AWGN, get_complex_noise_vector


class AWGNTests(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_000_setup(self):
        snr = 4.0
        effective_rate = 0.5
        subcarriers = 8
        snr_lin = effective_rate * 10. ** (snr / 10.)
        sigma = 1. / np.sqrt(1. * snr_lin * subcarriers)
        variance = sigma ** 2
        chan = AWGN(snr, effective_rate, subcarriers)
        self.assertAlmostEqual(chan.snr(), 4.)
        state = chan.state()
        self.assertAlmostEqual(state['snr'], snr)
        self.assertAlmostEqual(state['effective_rate'], effective_rate)
        self.assertEqual(state['subcarriers'], subcarriers)
        self.assertAlmostEqual(chan.sigma(), sigma)
        self.assertAlmostEqual(chan.variance(), variance)

    def test_001_status(self):
        chan = AWGN(4., 1., 1)
        self.assertEqual(chan.channel_length(), 1)
        self.assertIsNone(chan.channel_taps())
        self.assertIsNone(chan.channel_gains())

    def test_002_statistics(self):
        nlen = int(2 ** 17)
        variance = 0.7
        sigma = np.sqrt(variance)
        noise = get_complex_noise_vector(nlen, sigma)
        self.assertEqual(noise.dtype, np.complex64)
        self.assertEqual(noise.size, nlen)
        self.assertAlmostEqual(
            calculate_average_signal_energy(noise), variance, 1)

    def test_002_AWGN(self):
        snrs = np.arange(-4., 11.)
        vec_len = 500000
        for s in snrs:
            variance = 10. ** (-s / 10.)
            chan = AWGN(s)
            self.assertAlmostEqual(chan.variance(), variance, 14)
            t = np.zeros(vec_len, dtype=np.complex64)
            r = chan.transmit(t)
            chan.step()  # make sure this has no effect.
            e = calculate_average_signal_energy(r)
            # print(s, e, variance)
            self.assertTrue(np.abs(e - variance) < .05)

    def test_003_FFT(self):
        snrs = np.arange(-4., 11., 3)
        vec_len = 2 ** 16
        subcarriers = 8
        t = np.zeros(subcarriers, dtype=np.complex64)
        for s in snrs:
            variance = 10. ** (-s / 10.)
            chan = AWGN(s, subcarriers=subcarriers)
            self.assertAlmostEqual(chan.variance(), variance / subcarriers, 9)
            test_energy = np.zeros(vec_len)
            for v in range(vec_len):
                tr = chan.transmit(t)
                chan.step()
                fr = np.fft.fft(tr)
                te = calculate_average_signal_energy(tr)
                fe = calculate_average_signal_energy(fr)
                self.assertAlmostEqual(fe / te, subcarriers, 5)
                test_energy[v] = fe
            self.assertAlmostEqual(np.mean(test_energy), variance, 1)
