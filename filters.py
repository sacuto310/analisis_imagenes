import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


# Mean Filter
def meanFilter(img_data):    
    filtered_image_data = np.zeros_like(img_data)
    for x in range(1, img_data.shape[0]-2) :
        for y in range(1, img_data.shape[1]-2) :
            for z in range(1, img_data.shape[2]-2) :
                avg = 0
                for dx in range(-1, 1) :
                    for dy in range(-1, 1) :
                        for dz in range(-1, 1) :
                            avg = avg + img_data[x+dx, y+dy, z+dz]

                filtered_image_data[x+1, y+1, z+1] = avg / 27
    return filtered_image_data

# Median Filter
def medianFilter(image_data):
    filtered_image_data = np.zeros_like(image_data)
    for x in range(1, image_data.shape[0]-2) :
        for y in range(1, image_data.shape[1]-2) :
            for z in range(1, image_data.shape[2]-2) :
                neightbours = []
                for dx in range(-1, 1) :
                    for dy in range(-1, 1) :
                        for dz in range(-1, 1) :
                            neightbours.append(image_data[x+dx, y+dy, z+dz])

                median = np.median(neightbours)
                filtered_image_data[x+1, y+1, z+1] = median
    return filtered_image_data

def medianWBorders(image_data):
    # Median Filter with borders
    filtered_image_data = np.zeros_like(image_data)

    #threshold = 500

    # Estimate the standard deviation of the pixel intensity
    std = np.std(image_data)

    for x in range(1, image_data.shape[0]-2):
        for y in range(1, image_data.shape[1]-2):
            for z in range(1, image_data.shape[2]-2):
                # Compute the derivatives in x, y, and z directions
                dx = image_data[x+1, y, z] - image_data[x-1, y, z]
                dy = image_data[x, y+1, z] - image_data[x, y-1, z]
                dz = image_data[x, y, z+1] - image_data[x, y, z-1]

                # Compute the magnitude of the gradient
                magnitude = np.sqrt(dx*dx + dy*dy + dz*dz)

            
                # Compute the threshold using a fraction of the standard deviation
                threshold = 3 * std

                # If the magnitude is below the threshold, apply median filter
                if magnitude < threshold:
                    neighbours = []
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            for dz in range(-1, 2):
                                neighbours.append(image_data[x+dx, y+dy, z+dz])
                    median = np.median(neighbours)
                    filtered_image_data[x, y, z] = median
                else:
                    filtered_image_data[x, y, z] = image_data[x, y, z]
    return filtered_image_data