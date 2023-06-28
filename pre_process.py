import os
import streamlit as st
import tempfile
import nibabel as nib
import numpy as np
from skimage import exposure, filters
import matplotlib.pyplot as plt
from filters import meanFilter, medianFilter, medianWBorders
from standar import rescaling, z_score, white_stripe
import SimpleITK as sitk
import streamlit as st

def pre_process():
    st.title("Pre-Procesamiento")
    # Cargar archivo .nii o .nii.gz
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

        # Mostrar el menú desplegable de opciones de standarizacion y el botón de confirmación
        standardization_name = st.selectbox('Seleccione un metodo de estandarización', ['Ninguno', 'Rescaling', 'Z-score', 'White stripe'])
        if standardization_name != 'Ninguno':
            if st.button('Aplicar estandarizacion'):
                if standardization_name == 'Rescaling':
                    img_data = rescaling(img_data)
                elif standardization_name == 'Z-score':
                    img_data = z_score(img_data)
                elif standardization_name == 'White stripe':
                    img_data = white_stripe(img_data)

        # Mostrar el menú desplegable de opciones de filtros y el botón de confirmación
        filter_name = st.selectbox('Seleccione un filtro', ['Ninguno', 'Media', 'Mediana', 'Mediana con Bordes'])
        if filter_name != 'Ninguno':
            if st.button('Aplicar Filtro'):
                if filter_name == 'Media':
                    img_data = meanFilter(img_data)
                elif filter_name == 'Mediana':
                    img_data = medianFilter(img_data)
                elif filter_name == 'Mediana con Bordes':
                    img_data = medianWBorders(img_data)             

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