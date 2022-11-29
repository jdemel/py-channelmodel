#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019, 2020 Johannes Demel.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np
import matplotlib.pyplot as plt

from powerdelayprofile import PowerDelayProfile, FFTPowerDelayProfile
from powerdelayprofile import calculate_vector_signal_energy


def plot_power_delay_profile(rms_delay_spread=46.e-9,
                             max_delay_spread=250.e-9):
    for bandwidth in (20.e6, 50.e6, 100.e6):
        print(bandwidth)
        pdp = PowerDelayProfile(rms_delay_spread, max_delay_spread, bandwidth)

        s = pdp.supports()
        t = pdp.taps().real
        print(s)
        print(t)
        plt.plot(s, t, label='{:.0f}MHz'.format(bandwidth * 1e-6), marker='x')
    plt.xlabel('time delay [s]')
    plt.ylabel('amplitude')
    plt.legend()
    plt.grid()
    plt.show()


def plot_fft_power_delay_profile(rms_delay_spread=46.e-9,
                                 max_delay_spread=250.e-9, bandwidth=100.e6):
    for bandwidth in (20.e6, 50.e6, 100.e6):
        for subcarriers in (32, 64, 135):
            print(bandwidth)
            pdp = FFTPowerDelayProfile(rms_delay_spread, max_delay_spread,
                                       bandwidth, subcarriers)

            s = pdp.supports()
            t = pdp.taps().real

            f_taps = np.fft.fft(pdp.taps(), subcarriers)
            e = calculate_vector_signal_energy(f_taps)
            print('energy', e, pdp._scale)
            plt.plot(s, t, label='{:.0f}MHz, {}'.format(bandwidth * 1e-6,
                                                        subcarriers))
    plt.xlabel('time delay [s]')
    plt.ylabel('amplitude')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    # bandwidth = 20.e6
    # rms_channel_delay = 46.e-9
    # max_delay_spread = 250.e-9
    plot_fft_power_delay_profile()
    plot_power_delay_profile()


if __name__ == '__main__':
    main()
