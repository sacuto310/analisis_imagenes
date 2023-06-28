import numpy as np
import matplotlib.pyplot as plt


def threshold(image, tau, tol):
    while True:
        binary_image = image >= tau

        mBG = np.mean(image[binary_image == False])
        mFG = np.mean(image[binary_image == True])

        new_tau = 0.5 * (mBG + mFG)

        if abs(tau - new_tau) < tol:
            break

        tau = new_tau

    return binary_image

