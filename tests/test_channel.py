#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Johannes Demel.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
import unittest

from channelmodel.coherence import ChannelCoherenceRappaport
from channelmodel.powerdelayprofile import PowerDelayProfile, FFTPowerDelayProfile
from channelmodel.timevariantchannel import TimeVariantChannel
from channelmodel.timevariantchannel import CoherentTimeVariantChannel
from channelmodel.frequencydomainchannel import FrequencyDomainChannel
from channelmodel import ChannelFactory


class TimeVariantTests(unittest.TestCase):
    def setUp(self):
        self._coherence = ChannelCoherenceRappaport(3.8e9, 15.)
        self._pdp = PowerDelayProfile(46.8e-9, 250.e-9, 20.e6)

    def tearDown(self):
        pass

    def test_001_setup(self):
        chan = TimeVariantChannel(self._pdp)
        self.assertEqual(chan.channel_length(), self._pdp.num_taps())
        self.assertEqual(chan.channel_taps().size, self._pdp.num_taps())
        chan.step()
        self.assertEqual(chan.channel_taps().size, self._pdp.num_taps())

    def test_002_statistics(self):
        channel = TimeVariantChannel(self._pdp)
        num_iterations = 200000
        se = np.zeros(num_iterations)
        t0 = np.zeros(num_iterations, dtype=channel.channel_taps().dtype)
        for i in range(num_iterations):
            channel.step(1.e-3)
            t = channel.channel_taps()
            e = np.sum(np.abs(t) ** 2)
            se[i] = e
            t0[i] = t[0]
        self.assertAlmostEqual(np.mean(se), 1.0, 1)
        self.assertGreaterEqual(np.std(se), .8)
        self.assertGreaterEqual(np.var(se), .7)
        self.assertAlmostEqual(np.mean(t0.real), 0.0, 1)
        self.assertAlmostEqual(np.mean(t0.imag), 0.0, 1)
        ta = self._pdp.taps()[0].real * np.sqrt(1. / 2.)
        self.assertAlmostEqual(np.std(t0.real), ta, 1)
        self.assertAlmostEqual(np.std(t0.imag), ta, 1)

    def test_003_setup(self):
        chan = CoherentTimeVariantChannel(self._pdp, self._coherence)
        self.assertEqual(chan.channel_length(), self._pdp.num_taps())
        self.assertEqual(chan.channel_taps().size, self._pdp.num_taps())
        chan.step()
        self.assertEqual(chan.channel_taps().size, self._pdp.num_taps())
        s = chan.state()
        for k, v in self._coherence.state().items():
            self.assertAlmostEqual(s[k], v)
        print(chan.state())

    def test_004_statistics(self):
        channel = CoherentTimeVariantChannel(self._pdp, self._coherence)
        num_iterations = 200000
        se = np.zeros(num_iterations)
        t0 = np.zeros(num_iterations, dtype=channel.channel_taps().dtype)
        for i in range(num_iterations):
            channel.step(1.e-3)
            t = channel.channel_taps()
            e = np.sum(np.abs(t) ** 2)
            se[i] = e
            t0[i] = t[0]
        self.assertAlmostEqual(np.mean(se), 1.0, 1)
        self.assertGreaterEqual(np.std(se), .8)
        self.assertGreaterEqual(np.var(se), .7)
        self.assertAlmostEqual(np.mean(t0.real), 0.0, 1)
        self.assertAlmostEqual(np.mean(t0.imag), 0.0, 1)
        ta = self._pdp.taps()[0].real * np.sqrt(1. / 2.)
        self.assertAlmostEqual(np.std(t0.real), ta, 1)
        self.assertAlmostEqual(np.std(t0.imag), ta, 1)


class FrequencyDomainChannelTests(unittest.TestCase):
    def setUp(self):
        coherence = ChannelCoherenceRappaport(3.8e9, 15.)
        pdp = FFTPowerDelayProfile(46.8e-9, 250.e-9, 20.e6, 135)
        self._timevariantchannel = CoherentTimeVariantChannel(pdp, coherence)
        self._iterations = 100000

    def tearDown(self):
        pass

    def test_001_setup(self):
        pdp = PowerDelayProfile(46.8e-9, 250.e-9, 20.e6)
        time_variant_channel = TimeVariantChannel(pdp)
        self.assertRaises(KeyError, FrequencyDomainChannel,
                          time_variant_channel)
        chan = FrequencyDomainChannel(self._timevariantchannel)
        self.assertEqual(chan.subcarriers(), 135)
        self.assertEqual(chan.freq_domain_taps().size, 135)
        self.assertEqual(chan.freq_domain_gains().size, 135)
        chan.step()
        self.assertEqual(chan.freq_domain_taps().size, 135)
        self.assertEqual(chan.freq_domain_gains().size, 135)

    def test_002_statistics(self):
        chan = FrequencyDomainChannel(self._timevariantchannel)
        gdtype = chan.freq_domain_gains().dtype
        gains = np.zeros((self._iterations, chan.subcarriers()), dtype=gdtype)
        for i in range(self._iterations):
            chan.step()
            gains[i] = chan.freq_domain_gains()

        mgains = np.mean(gains, axis=0)
        self.assertTrue(np.all(np.abs(mgains - 1.) < 3.e-2))


class ChannelFactoryTests(unittest.TestCase):
    def setUp(self):
        self._rms_delay_spread = 46.8e-9
        self._max_delay_spread = 250.e-9
        self._channel_domain = 'time'
        self._channel_type = 'rayleigh'
        self._effective_rate = 1.

    def tearDown(self):
        pass

    def test_001_create(self):
        for txa in range(1, 5):
            for rxa in range(1, 5):
                cfac = ChannelFactory(self._channel_domain, self._channel_type, self._effective_rate,
                                      rms_delay_spread=self._rms_delay_spread, max_delay_spread=self._max_delay_spread,
                                      tx_antennas=txa, rx_antennas=rxa, equalizer_type='ZF')
                chan = cfac.create(13.)
                state = chan.state()
                print(state)
                self.assertAlmostEqual(chan.snr(), 13.)
                self.assertAlmostEqual(
                    state['effective_rate'], self._effective_rate)
                self.assertAlmostEqual(
                    state['rms_delay_spread'], self._rms_delay_spread)
                self.assertAlmostEqual(
                    state['max_delay_spread'], self._max_delay_spread)
                self.assertAlmostEqual(
                    state['scale'], 1. / np.sqrt(1. * txa * rxa))
                self.assertEqual(chan.tx_antennas(), txa)
                self.assertEqual(chan.rx_antennas(), rxa)
                self.assertEqual(chan.channel_dimensions(), (rxa, txa))
                self.assertEqual(np.shape(chan.channel_taps())
                                 [0:2], (rxa, txa))

    def test_002_normalization(self):
        for txa in range(1, 5):
            for rxa in range(1, 5):
                cfac = ChannelFactory(self._channel_domain, self._channel_type, self._effective_rate,
                                      rms_delay_spread=self._rms_delay_spread, max_delay_spread=self._max_delay_spread,
                                      tx_antennas=txa, rx_antennas=rxa, equalizer_type='ZF')
                chan = cfac.create(50.)
                fadings = chan._fading_channels

                time_channels = [i for row in fadings for i in row]
                self.assertEqual(len(time_channels), txa * rxa)

                for c in time_channels:
                    pdp = c._time_variant_channel._pdp
                    self.assertAlmostEqual(1. / np.sqrt(txa * rxa), pdp._scale)
                    energy = np.sum(np.abs(pdp.taps()) ** 2)
                    self.assertAlmostEqual(energy, 1. / txa / rxa)


if __name__ == '__main__':
    unittest.main(failfast=True)
