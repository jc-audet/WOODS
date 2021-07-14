
import os 
import numpy as np
from scipy import fft

import matplotlib.pyplot as plt


# SAMPLE_RATE = 200  # Hertz
# DURATION = 5  # Seconds

## Triangle frequency shape
yf = np.zeros(1000)
yf[600:900] = np.linspace(0, 500, num=300)
yf[400] = 200

plt.figure()
plt.plot(np.abs(yf))
plt.title("Label 0 Fourier domain")

new_sig = fft.irfft(yf, n=100000)

plt.figure()
plt.plot(new_sig[50000:55000] / np.max(new_sig[50000:55000]))
plt.title("Label 0 time series")

## Rectangle frequency shape
yf = np.zeros(1000)
yf[600:900] = np.linspace(500, 0, num=300)
yf[50] = 200

plt.figure()
plt.plot(np.abs(yf))
plt.title("Label 1 Fourier domain")

new_sig = fft.irfft(yf, n=100000)

plt.figure()
plt.plot(new_sig[50000:55000] / np.max(new_sig[50000:55000]))
plt.title("Label 1 time series")

plt.show()