import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy import stats as st
import statistics as stat

# Rescaling
def rescaling(image_data):
    min_value = image_data.min()
    max_value = image_data.max()
    image_data_rescaled = (image_data - min_value) / (max_value - min_value)
    plt.hist(image_data_rescaled[image_data_rescaled>0.01].flatten(), 100)
    return image_data_rescaled

# Z-score
def z_score(image_data):
    mean_value = image_data[image_data > 10].mean()
    standard_deviation_value = image_data[image_data > 10].std()

    print(mean_value)
    print(standard_deviation_value)

    image_data_rescaled = (image_data - mean_value) / (standard_deviation_value)
    plt.hist(image_data_rescaled[image_data_rescaled>0.01].flatten(), 100)
    return image_data_rescaled

# White stripe
def white_stripe(image_data):    
    # Calcular el histograma
    hist, bin_edges = np.histogram(image_data.flatten(), bins=100)

    # Encontrar los picos del histograma
    picos, _ = find_peaks(hist, height=100)
    val_picos=bin_edges[picos]
    print(val_picos[1])

    # Imagen reecalada
    image_data_rescaled=image_data/val_picos[1]

    # Mostrar el histograma con los picos identificados
    plt.axvline(val_picos[4], color='r', linestyle='--')
    plt.hist(image_data[image_data>10].flatten(), bins=100)
    plt.plot(bin_edges[picos], hist[picos], "x")
    plt.show()
    return image_data_rescaled