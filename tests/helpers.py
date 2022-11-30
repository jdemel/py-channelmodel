#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Johannes Demel.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import numpy as np


def generate_random_qpsk(n_syms):
    b = np.random.randint(0, 2, (2, n_syms))
    b = 1. - 2. * b
    return (1. * b[0] + 1.j * b[1]) * np.sqrt(1. / 2.)


def calculate_average_signal_energy(s):
    return np.mean(s.real ** 2 + s.imag ** 2)
