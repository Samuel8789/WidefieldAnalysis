"""
This module contains entry points to analyses functions to
ease development process. It is strongly suggested to use this as an entry point only,
and to always work in the scope of interest via debuggers.
"""
import os
import matplotlib.pyplot as plt

# import gui.analyses
# from gui.analyses import CalciumImaging, RetinotopicMapper
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import rfft, rfftfreq
import numpy.typing as npt
import math

os.chdir(r"C:/Users/sp3660/Documents/Github/WidefieldAnalysis")
# calcium_imaging = CalciumImaging(
#     experiment_directory=os.path.join(os.getcwd(), 'retinotopic_mapping')
# )
# calcium_process_stack = calcium_imaging.process_stack()

# %%

# retinotopic_mapping = RetinotopicMapper(
#     ttl_directory=os.path.join(os.getcwd(), 'retinotopic_mapping', 'ttl_output'),
#     wfield_directory=os.path.join(os.getcwd(), 'retinotopic_mapping')
# )
#
# cleaned_data = retinotopic_mapping.generate_maps()  # data for sign maps development

# %% SIMPLE DEVELOPMENT DATA FOR SIGN MAP DEVELOPMENT

from scipy.stats import zscore

import pickle

with open(r'C:/Users/sp3660/Documents/Github/WidefieldAnalysis/data_to_signmap_dev_UMBERTO.pickle', 'rb') as f:
    data = pickle.load(f)

stack = data['stack']
direction_order = data['direction_order']
direction_length = data['direction_length']
height = data['height']
width = data['width']
stack_time_aligned = data['stack_time_aligned']
#phase_map_smoothing_factor = data['phase_map_smoothing_factor']
phase_map_smoothing_factor = 5
stimon_info = data['stimon_info']
stimulus_line = data['stimulus_line']
blue_line = data['blue_line']
violet_line = data['violet_line']
camera_sampling_rate = data['camera_sampling_rate']
analog_sampling_rate = data['analog_sampling_rate']
kernel_smoothing_factor = data['kernel_smoothing_factor']
sweep_rate = data['sweep_rate']


# you cannot apply zscore, np.mean and other of these to a list. stack_time_aligned is a list of matrices, one for
# each direction of sweeping. if you want to apply any functions to each matrix, you must use a for loop

#%%
output_for_each_matrix = []
for count, (ord, mat) in enumerate(zip(direction_order, stack_time_aligned)):
    stack_mean = zscore(np.mean(mat, axis=(1, 2)))

    plt.imshow(mat[0,:,:], cmap = 'jet')
    plt.colorbar()
    plt.show()

    aaa = np.mean(mat, axis=0)
    plt.imshow(aaa, cmap = 'jet')
    plt.colorbar()
    plt.show()

    outliers = stack_mean > 2

    mat = mat[~outliers, :, :]

    freqs: npt.NDArray = rfftfreq(mat.shape[0], (1 / camera_sampling_rate))

    fourier_transf: npt.NDArray = rfft(mat, axis=0)

    deviations: npt.NDArray = np.abs(freqs - sweep_rate[ord])

    nearest_maps: npt.NDArray = fourier_transf[np.argmin(deviations), :, :]
    magnitude_map: np.float64 = np.absolute(nearest_maps)
    phase_map: np.float64 = np.angle(nearest_maps)

    plt.imshow(phase_map, cmap = 'jet')
    plt.colorbar()
    plt.show()

    smoothed_phase_map = gaussian_filter(phase_map, phase_map_smoothing_factor)

    output_for_each_matrix.append(smoothed_phase_map)


    plt.imshow(smoothed_phase_map, cmap = 'jet')
    plt.colorbar()
    plt.show()

    # plt.imshow(output_for_each_matrix[3], cmap = 'jet')
    # plt.colorbar()
    # plt.show()

# output_for_each_matrix will be a list of smoothed_phase_map, one for each matrix contained in stack_time_aligned,
# then one for each direction of sweeping

# def _generate_sign_map(
#         smoothed_phase_maps: dict[str, npt.NDArray],
#         phase_map_smoothing_factor
# ) -> dict[str, npt.NDArray]:
#
#     azimuth_map = smoothed_phase_maps["top2bottom"] - smoothed_phase_maps["bottom2top"]
#     altitude_map = smoothed_phase_maps["left2right"] - smoothed_phase_maps["right2left"]
#
#     grad1a, grad1b = np.gradient(altitude_map)
#     grad2a, grad2b = np.gradient(azimuth_map)
#
#     grad1 = grad1a + 1j * grad1b
#     grad2 = grad2a + 1j * grad2b
#
#     sign_map = np.sin(np.angle(grad1) - np.angle(grad2))
#     sign_map = gaussian_filter(sign_map, phase_map_smoothing_factor)
#     return sign_map
#
#
#     plt.imshow(sign_map, cmap='jet')
#     plt.colorbar()
#     plt.show()
#
#


for frame in output_for_each_matrix:
    plt.imshow(frame, cmap = 'jet')
    plt.colorbar()
    plt.show()




azimuth_map = output_for_each_matrix[1] - output_for_each_matrix[2]
altitude_map = output_for_each_matrix[0] - output_for_each_matrix[3]

plt.imshow(azimuth_map, cmap='jet')
plt.colorbar()
plt.show()

plt.imshow(altitude_map, cmap='jet')
plt.colorbar()
plt.show()

aaaa = azimuth_map + altitude_map
plt.imshow(aaaa, cmap='jet')
plt.colorbar()
plt.clim(-4, 4);
plt.show()


grad1a, grad1b = np.gradient(altitude_map)
grad2a, grad2b = np.gradient(azimuth_map)

grad1 = grad1a + 1j * grad1b
grad2 = grad2a + 1j * grad2b


sign_map = np.sin(np.angle(grad1) - np.angle(grad2))

plt.imshow(sign_map, cmap = 'jet')
plt.clim(-2, 2);
plt.show()







#sign_map = gaussian_filter(sign_map, phase_map_smoothing_factor)

grad1_or = np.gradient(altitude_map)
grad2_or = np.gradient(azimuth_map)

graddir1 = np.zeros(np.shape(grad1_or[0]))
graddir2 = np.zeros(np.shape(grad2_or[0]))

for i in range(altitude_map.shape[0]):
    for j in range(azimuth_map.shape[1]):
        graddir1[i,j] = math.atan2(grad1_or[1][i,j], grad1_or[0][i,j])
        graddir2[i,j] = math.atan2(grad2_or[1][i,j], grad2_or[0][i,j])

diff = np.multiply(np.exp(1j * graddir1), np.exp(-1j * graddir2))

map_original = np.sin(np.angle(diff))

plt.imshow(map_original)
plt.show()


plt.imshow(np.angle(grad2), cmap = 'jet')
plt.show()

plt.imshow(sign_map, cmap = 'jet')
plt.show()