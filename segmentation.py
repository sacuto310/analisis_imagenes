import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy import stats as st
import statistics as stat

def tresholding(image_data,tau,tol):

        print ("se pudo")
        #plt.imshow(self.image_data[:, :, 100])

        while True:
            #print(tau)

            segmentationr = image_data >= tau
            mBG = image_data[np.multiply(image_data > 10, segmentationr == 0)].mean()
            mFG = image_data[np.multiply(image_data > 10, segmentationr == 1)].mean()

            tau_post = 0.5 * (mBG + mFG)

            if np.abs(tau - tau_post) < tol:
                break
            else:
             tau = tau_post
        print(tau)
        print(segmentationr.shape)
        return segmentationr

def growing(image, tol=50, x=100, y=100, z=20):
        # Region Growing
        segmentation = np.zeros_like(image)
        if segmentation[x,y,z] == 1:
            return
        valor_medio_cluster = image[x,y,z]
        segmentation[x,y,z] = 1
        vecinos = [(x, y, z)]
        while vecinos:
            x, y, z = vecinos.pop()
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    for dz in [-1,0,1]:
                        #vecino
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if nx >= 0 and nx < image.shape[0] and \
                            ny >= 0 and ny < image.shape[1] and \
                            nz >= 0 and nz < image.shape[2]:
                            if np.abs(valor_medio_cluster - image[nx,ny,nz]) < tol and \
                                segmentation[nx,ny,nz] == 0:
                                segmentation[nx,ny,nz] = 1
                                vecinos.append((nx, ny, nz))
        return segmentation

def k_means(image, ks,iteracion):
        
        # Inicialización de valores k
        k_values = np.linspace(np.amin(image), np.amax(image), ks)
        iteracion=10
        for i in range(iteracion):
            d_values = [np.abs(k - image) for k in k_values]
            segmentationr = np.argmin(d_values, axis=0)

            for k_idx in range(ks):
                k_values[k_idx] = np.mean(image[segmentationr == k_idx])

        return segmentationr

def GMM(image_data):
        w1 = 1/3
        w2 = 1/3
        w3 = 1/3
        mu1 = 0
        sd1 = 50
        mu2 = 100
        sd2 = 50
        mu3 = 150
        sd3 = 50

        seg = np.zeros_like(image_data)
        for iter in range(1, 5) :

            # Compute likelihood of belonging to a cluster
            p1 = 1/np.sqrt(2*np.pi*sd1*2) * np.exp(-0.5*np.power(image_data - mu1, 2) / sd1*2)
            p2 = 1/np.sqrt(2*np.pi*sd2*2) * np.exp(-0.5*np.power(image_data - mu2, 2) / sd2*2)
            p3 = 1/np.sqrt(2*np.pi*sd3*2) * np.exp(-0.5*np.power(image_data - mu3, 2) / sd3*2)

            # Normalise probability
            r1 = np.divide(w1 * p1, w1 * p1 + w2 * p2 + w3 * p3)
            r2 = np.divide(w2 * p2, w1 * p1 + w2 * p2 + w3 * p3) 
            r3 = np.divide(w3 * p3, w1 * p1 + w2 * p2 + w3 * p3) 

            # Update parameters
            w1 = r1.mean()
            w2 = r2.mean()
            w3 = r3.mean()
            mu1 = np.multiply(r1, image_data).sum() / r1.sum()
            sd1 = np.sqrt(np.multiply(r1, np.power(image_data - mu1, 2)).sum() / r1.sum())
            mu2 = np.multiply(r2, image_data).sum() / r2.sum()
            sd2 = np.sqrt(np.multiply(r2, np.power(image_data - mu2, 2)).sum() / r2.sum())
            mu3 = np.multiply(r3, image_data).sum() / r3.sum()
            sd3 = np.sqrt(np.multiply(r3, np.power(image_data - mu3, 2)).sum() / r3.sum())

        # Perform segmentation
        seg[np.multiply(r1 > r2, r1 > r3)] = 0
        seg[np.multiply(r2 > r1, r2 > r3)] = 1
        seg[np.multiply(r3 > r1, r3 > r2)] = 2
        
        return seg