import os
import streamlit as st
import tempfile
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import streamlit as st
from skimage import exposure, filters

def previsualize():
    st.title("Pre-visualizacion de la imagen")
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

            # Crear botones redondos para seleccionar el eje
            axis = st.radio("Seleccionar eje", ["X", "Y", "Z"])

            # Seleccionar la sección de la imagen correspondiente al eje seleccionado
            if axis == "X":
                slice_idx = st.slider("Seleccionar slice", 0, img_data.shape[0] - 1, img_data.shape[0] // 2)
                img_slice = img_data[slice_idx, :, :]
            elif axis == "Y":
                slice_idx = st.slider("Seleccionar slice", 0, img_data.shape[1] - 1, img_data.shape[1] // 2)
                img_slice = img_data[:, slice_idx, :]
            else:
                slice_idx = st.slider("Seleccionar slice", 0, img_data.shape[2] - 1, img_data.shape[2] // 2)
                img_slice = img_data[:, :, slice_idx]
                            
            # Normalizar los valores de píxeles para que estén dentro del rango [0, 1]
            img_slice = exposure.rescale_intensity(img_slice, in_range='image', out_range=(0, 1))

            # Mostrar la imagen resultante
            st.image(img_slice, width=300)
