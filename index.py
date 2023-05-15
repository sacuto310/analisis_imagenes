import os
import streamlit as st
import tempfile
import nibabel as nib
import numpy as np
from skimage import exposure, filters
import matplotlib.pyplot as plt
from filters import meanFilter, medianFilter, medianWBorders

def main():
    st.set_page_config(page_title="Proyecto Procesamiento de imágenes", layout="wide")

    menu = ["Pre-visualizacion de la imagen", "Visualizar volumenes", "Pre-Procesamiento"]
    choice = st.sidebar.selectbox("Seleccionar página", menu)

    if choice == "Pre-visualizacion de la imagen":
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

    elif choice == "Visualizar volumenes":
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
            slice_idx = st.slider('Seleccionar un eje', 0, img_data.shape[0]-1, img_data.shape[0]//2)
            update_view(slice_idx)

            # Eliminar el archivo temporalmente guardado
            os.remove(temp_file.name)
    
    elif choice == "Pre-Procesamiento":
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
            slice_idx = st.slider('Seleccionar un eje', 0, img_data.shape[0]-1, img_data.shape[0]//2)
            update_view(slice_idx)

            # Eliminar el archivo temporalmente guardado
            os.remove(temp_file.name)

if __name__ == "__main__":
    main()
