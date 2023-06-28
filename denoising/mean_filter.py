import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def mean_filter_image(image_data):
    filtered_image = np.zeros_like(image_data)
    for x in range (0, image_data.shape[0]-2):
        for y in range(0, image_data.shape[1]-2):
            for z in range(0, image_data.shape[2]-2):
                # voxel_int = image_data[x,y,z]

                avg = 0

                for dx in range(-1,1) :
                    for dy in range(-1,1) :
                        for dz in range(-1,1) :
                            avg = avg + image_data[x+dx, y+dy, z+dz]

                filtered_image[x,y,z] = avg / 3 ** 3
    return  filtered_image

def mean_filter_with_edges(image_data):
    filtered_image = np.zeros_like(image_data)
    dfdx = np.zeros_like(image_data)
    dfdy = np.zeros_like(image_data)
    dfdz = np.zeros_like(image_data)

    # Calculate gradients for edge handling
    for x in range(1, image_data.shape[0]-1):
        for y in range(1, image_data.shape[1]-1):
            for z in range(1, image_data.shape[2]-1):
                dfdx[x, y, z] = image_data[x+1, y, z] - image_data[x-1, y, z]
                dfdy[x, y, z] = image_data[x, y+1, z] - image_data[x, y-1, z]
                dfdz[x, y, z] = image_data[x, y, z+1] - image_data[x, y, z-1]

    # Apply mean filter with edge handling
    for x in range(1, image_data.shape[0]-1):
        for y in range(1, image_data.shape[1]-1):
            for z in range(1, image_data.shape[2]-1):
                avg = 0
                count = 0

                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        for dz in range(-1, 2):
                            avg += image_data[x+dx, y+dy, z+dz]
                            count += 1

                # Adjust average for edge pixels
                if x == 1:
                    avg += dfdx[x, y, z]
                    count += 1
                if y == 1:
                    avg += dfdy[x, y, z]
                    count += 1
                if z == 1:
                    avg += dfdz[x, y, z]
                    count += 1

                filtered_image[x, y, z] = avg / count

    return filtered_image