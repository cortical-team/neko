#! /usr/bin/env python

# This file is part of the SpikeCoding repository - MAVLab TU Delft
#
# MIT License
#
# Copyright (c) 2021 Julien Dupeyroux
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# @author Julien Dupeyroux, Stein Stroobants

import numpy as np
from scipy.stats import norm


def temporal_contrast(data, factor):
    # Based on algorithm provided in:
    #   Sengupta et al. (2017)
    #   Petro et al. (2020)
    diff = np.zeros(len(data) - 1)
    spikes = np.zeros(len(data))
    for i in range(len(data) - 1):
        diff[i] = data[i + 1] - data[i]
    threshold = np.mean(diff) + factor * np.std(diff)
    diff = np.insert(diff, 0, diff[1])
    for i in range(len(data)):
        if diff[i] > threshold:
            spikes[i] = 1
        elif diff[i] < -threshold:
            spikes[i] = -1
    return spikes, threshold


def step_forward(data, threshold):
    # Based on algorithm provided in:
    #   Petro et al. (2020)
    startpoint = data[0]
    spikes = np.zeros(len(data))
    base = startpoint
    for i in range(1, len(data)):
        if data[i] > base + threshold:
            spikes[i] = 1
            base = base + threshold
        elif data[i] < base - threshold:
            spikes[i] = -1
            base = base - threshold
    return spikes, startpoint


def moving_window(data, threshold, window):
    # Based on algorithm provided in:
    #   Petro et al. (2020)
    startpoint = data[0]
    spikes = np.zeros(len(data))
    base = np.mean(data[0:window + 1])
    for i in range(window + 1):
        if data[i] > base + threshold:
            spikes[i] = 1
        elif data[i] < base - threshold:
            spikes[i] = -1
    for i in range(window + 2, len(data)):
        base = np.mean(data[(i - window - 1):(i - 1)])
        if data[i] > base + threshold:
            spikes[i] = 1
        elif data[i] < base - threshold:
            spikes[i] = -1
    return spikes, startpoint


def hough_spike(data, fir):
    # Based on algorithm provided in:
    #   Schrauwen et al. (2003)
    spikes = np.zeros(len(data))
    shift = min(data)
    data = data - shift * np.ones(len(data))
    for i in range(len(data)):
        count = 0
        for j in range(len(fir)):
            if i + j < len(data):
                if data[i + j] >= fir[j]:
                    count = count + 1
        if count == len(fir):
            spikes[i] = 1
            for j in range(len(fir)):
                if i + j < len(data):
                    data[i + j] = data[i + j] - fir[j]
    return spikes, shift


def modified_hough_spike(data, fir, threshold):
    # Based on algorithm provided in:
    #   Schrauwen et al. (2003)
    spikes = np.zeros(len(data))
    shift = min(data)
    data = data - shift * np.ones(len(data))
    for i in range(len(data)):
        error = 0
        for j in range(len(fir)):
            if i + j < len(data):
                if data[i + j] < fir[j]:
                    error = error + fir[j] - data[i + j]
        if error <= threshold:
            spikes[i] = 1
            for j in range(len(fir)):
                if i + j < len(data):
                    data[i + j] = data[i + j] - fir[j]
    return spikes, shift


def ben_spike(data, fir, threshold):
    # Based on algorithm provided in:
    #   Petro et al. (2020)
    #   Sengupta et al. (2017)
    #   Schrauwen et al. (2003)
    spikes = np.zeros(len(data))
    shift = min(data)
    data = data - shift * np.ones(len(data))
    for i in range(len(data) - len(fir) + 1):
        err1 = 0
        err2 = 0
        for j in range(len(fir)):
            err1 = err1 + abs(data[i + j] - fir[j])
            err2 = err2 + abs(data[i + j - 1])
        if err1 <= err2 * threshold:
            spikes[i] = 1
            for j in range(len(fir)):
                if i + j + 1 < len(data):
                    data[i + j + 1] = data[i + j + 1] - fir[j]
    return spikes, shift


def grf_spike(data, m, min_input, max_input):
    # Adapted from algorithm provided in:
    #   Bohté et al. (2002)
    # Modifications: definition of sigma, removal of beta constant,
    #                and modified WTA process

    if np.isscalar(data):
        data = [data]

    spikes = np.zeros((len(data), m))
    neuron_outputs = np.zeros(m)

    for j in range(len(data)):
        for i in range(m):
            mu = min_input + (2 * (i + 1) - 3) / 2 * (max_input - min_input) / (m - 2)
            sigma = (max_input - min_input) / (m - 2)
            neuron_outputs[i] = norm.pdf(data[j], mu, sigma)

        spikes[j, np.argmax(neuron_outputs)] = 1
    return spikes


def one_hot_place_spike(data, m, min_input, max_input):
    # Simple population coding algorithm adapted from Stagsted et al. (2020) that represents inputs by a location. 
    # An input is assigned to the neuron that is closest to its value. 
    # Only one neuron fires at every timestep

    if np.isscalar(data):
        data = [data]

    spikes = np.zeros((len(data), m))

    for j in range(len(data)):
        size_change = 1 / 2 * (max_input - min_input) / (
                    m - 2)  # to make sure it has the same lower/upper bounds as the Bohte paper
        idx = int(np.round(
            ((data[j] - (min_input - size_change)) / ((max_input + size_change) - (min_input - size_change))) * (
                        m - 1)))
        spikes[j, idx] = 1

    return spikes


def grf_spike_with_internal_timesteps(data, min_input, max_input, neurons=10, timesteps=10, beta=1.5):
    """Create a series of spikes based on Gaussian Receptive Fields
    Adapted from algorithm provided in:
        Bohté et al. (2002)
    
    Keyword arguments:
    data -- 
    neurons -- numbers of neurons (default 10)
    timesteps -- number of timesteps (default 10)
    min_input -- minimal value
    max_input -- maximum value
    beta -- tuning parameter that determines the width of the receptive fields
    """

    if np.isscalar(data):
        data = [data]

    spikes = np.zeros((len(data), timesteps, neurons))
    responses = np.zeros(neurons)

    # Calculation of mu and sigma of the Gaussian receptive fields
    mu = min_input + (2 * (np.arange(neurons) + 1) - 3) / 2 * (max_input - min_input) / (neurons - 2)
    sigma = 1 / beta * (max_input - min_input) / (neurons - 2)
    max_prob = norm.pdf(mu[0], mu[0], sigma)

    for j in range(len(data)):
        for i in range(neurons):
            responses[i] = norm.pdf(data[j], mu[i], sigma)
            size_change = max_prob / (2 * timesteps)
            new = int(np.round(((responses[i] + size_change) / (max_prob + 2 * size_change) * (
                        timesteps + 1)) + 0.0001))  # 0.0001 for roundoff errors...
            spiking_time = timesteps - new
            if spiking_time < timesteps - 1:
                spikes[j, spiking_time, i] = 1
    spikes = spikes.reshape([len(data) * timesteps, neurons])
    return spikes
