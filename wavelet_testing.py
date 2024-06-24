# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:33:15 2024

@author: sp3660
"""

#%% 
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Sample signal generation (Replace this with your actual signal)
t = np.linspace(0, 1, 1000, endpoint=False)
A=2
signal = A*np.sin(2 * np.pi * 7 * t) + A*np.cos(2 * np.pi * 15 * t)

# Perform wavelet transform
wavelet = 'cmor1.5-1.0'  # Choose a wavelet, here 'morl' is used as an example
scales = np.arange(1, 128)
coefficients, frequencies = pywt.cwt(signal, scales, wavelet,sampling_period=1)

# Find the index corresponding to the maximum amplitude
max_index = np.unravel_index(np.argmax(np.abs(coefficients)), coefficients.shape)

# Extract frequency and phase of the signal component with the highest amplitude
max_frequency = frequencies[max_index[0]]
max_phase = np.angle(coefficients[max_index])

# Generate the sine wave with the extracted frequency and phase
generated_signal = np.sin(2 * np.pi * 5 * t + max_phase)

# Create figure and axes
fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)


ax3.plot(t,  A*np.sin(2 * np.pi * 7 * t), 'b', label='Original Signal')
ax3.plot(t,  A*np.cos(2 * np.pi * 15 * t), 'b', label='Original Signal')
ax3.plot(t, generated_signal, 'r--', label='Extracted Signal')
ax3.set_ylabel('Signal')
ax3.legend()
ax3.set_title('Original Signal and Extracted Signal')

# Plot original signal
ax1.plot(t, signal, 'b', label='Original Signal')
ax1.plot(t, generated_signal, 'r--', label='Extracted Signal')
ax1.set_ylabel('Signal')
ax1.legend()
ax1.set_title('Original Signal and Extracted Signal')

# Plot amplitude-time plot of the wavelet (scalogram)
extent = [t[0], t[-1], frequencies[-1], frequencies[0]]
img = ax2.imshow(np.abs(coefficients), extent=extent, cmap='jet', aspect='auto')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_xlabel('Time')
ax2.set_xlim(t[0], t[-1])

# Add colorbar below the top subplot
cbar_top = fig.colorbar(img, ax=ax1, orientation='horizontal', pad=0.05)
cbar_top.set_label('Magnitude')

# Add colorbar above the bottom subplot
cbar_bottom = fig.colorbar(img, ax=ax2, orientation='horizontal', pad=0.05)
cbar_bottom.set_label('Magnitude')

plt.show()
