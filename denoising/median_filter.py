import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def median_filter(image_data):
    filtered_image = np.zeros_like(image_data)

    for x in range(1, image_data.shape[0]-1):
        for y in range(1, image_data.shape[1]-1):
            for z in range(1, image_data.shape[2]-1):
                # Extraer la vecindad 3x3x3
                neighborhood = image_data[x-1:x+2, y-1:y+2, z-1:z+2]
                
                # Calcular la mediana de la vecindad
                median_value = np.median(neighborhood)
                
                # Asignar el valor mediano al píxel filtrado
                filtered_image[x, y, z] = median_value

    return filtered_image

def medianFilterBorders (image):
  # Median Filter with borders
	threshold = 2500
	filtered_image = np.zeros_like(image)

	for x in range(1, image.shape[0] - 2):
		for y in range(1, image.shape[1] - 2):
			for z in range(1, image.shape[2] - 2):
        # Compute the derivatives in x, y, and z directions
				dx = image[x + 1, y, z] - image[x - 1, y, z]
				dy = image[x, y + 1, z] - image[x, y - 1, z]
				dz = image[x, y, z + 1] - image[x, y, z - 1]

        # Compute the magnitude of the gradient
				magnitude = np.sqrt(dx * dx + dy * dy + dz * dz)

                    # Separate pixels based on the current threshold
				below_threshold = magnitude[magnitude < threshold]
				above_threshold = magnitude[magnitude >= threshold]
				threshold = (np.mean(below_threshold) + np.mean(above_threshold)) / 2

				if magnitude < threshold:
					neighbours = []
					for dx in range(-1, 2):
						for dy in range(-1, 2):
							for dz in range(-1, 2):
								neighbours.append(image[x + dx, y + dy, z + dz])
					median = np.median(neighbours)
					filtered_image[x, y, z] = median
				else:
					filtered_image[x, y, z] = image[x, y, z]
	return filtered_image