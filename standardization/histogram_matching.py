import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def histogram_matching(objective_data, origin_data,k):
    # Reshape the data arrays to 1D arrays
    objective_flat = objective_data.get_fdata().flatten()
    origin_flat = origin_data.get_fdata().flatten()


    reference_landmarks = np.percentile(objective_flat, np.linspace(0, 100, k))
    transform_landmarks = np.percentile(origin_flat, np.linspace(0, 100, k))

    piecewise_func = np.interp(origin_flat, transform_landmarks, reference_landmarks)


    transformed_data = piecewise_func.reshape(origin_data.shape)

    return transformed_data