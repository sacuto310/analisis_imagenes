import os
import streamlit as st
import tempfile
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import streamlit as st
from skimage import exposure, filters

def visualize():
    st.title("Visualizar volumenes")
    uploaded_file = st.file_uploader("Cargar archivo .nii o .nii.gz", type=["nii", "nii.gz"])
    if uploaded_file is not None:
        # Guarda el archivo cargado temporalmente con extensión .nii
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file.flush()
            temp_file.close()

        # Lee el archivo temporalmente guardado
        nii_img = nib.load(temp_file.name)

        # Obtener datos de píxeles de la imagen
        img_data = nii_img.get_fdata()

        # Normalizar los valores de píxeles para que estén dentro del rango [0, 1]
        img_data = exposure.rescale_intensity(img_data, in_range='image', out_range=(0, 1))

        # Crear figura con tres sub-figuras, una para cada eje
        fig, axs = plt.subplots(ncols=3, figsize=(15, 5))

        # Función para actualizar la visualización de la imagen según el valor del slider
        def update_view(value):
            axs[0].imshow(img_data[value, :, :])
            axs[1].imshow(img_data[:, value, :])
            axs[2].imshow(img_data[:, :, value])
            st.pyplot(fig)

        # Slider para navegar por los ejes de la imagen
        slice_idx = st.slider('Seleccionar un eje', 0, img_data.shape[2]-1, img_data.shape[2]//2)
        update_view(slice_idx)

        # Eliminar el archivo temporalmente guardado
        os.remove(temp_file.name)