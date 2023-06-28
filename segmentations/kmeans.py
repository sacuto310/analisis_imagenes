import numpy as np
import matplotlib.pyplot as plt


def kmeans_segmentation (image, iterations, ks ):
	# Inicialización de valores k
	k_values = np.linspace(np.amin(image), np.amax(image), ks)
	for i in range(iterations):
		d_values = [np.abs(k - image) for k in k_values]
		segmentationr = np.argmin(d_values, axis=0)

		for k_idx in range(ks):
			k_values[k_idx] = np.mean(image[segmentationr == k_idx])

	return segmentationr
