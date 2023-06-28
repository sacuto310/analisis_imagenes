import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
from standardization.zscore import zscore_image
from denoising.median_filter import medianFilterBorders

def estandarizar ():
    st.title("Preprocesar")

    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    with st.container():

        T1_image = None
        FLAIR_image = None
        IR_image = None

        input_files = st.file_uploader('Ingresa las imágenes del paciente', accept_multiple_files=True)

        # Procesar cada archivo subido
        file_paths = []
        if input_files:
            for file in input_files:
                file_name = file.name
                file_path = os.path.join('uploads', file_name)
                with open(file_path, 'wb') as f:
                    f.write(file.getbuffer())
                file_paths.append(file_path)

        # Mostrar los paths de los archivos subidos
        for path in file_paths:
            if "T1" in path :
                T1_image = path
            if "FLAIR" in path :
                FLAIR_image = path
            if "IR" in path :
                IR_image = path
        
        if T1_image and FLAIR_image and IR_image != None: 
            st.title("Preprocesar")
            image_T1 = nib.load(T1_image)
            image_T1_data = image_T1.get_fdata()
            image_IR = nib.load(IR_image)
            image_IR_data = image_IR.get_fdata()
            
            ### ESTANDARIZAR IMAGEN
            Estandarize_t1 = zscore_image(image_T1_data)
            Estandarize_ir = zscore_image(image_IR_data)

            ### REMCION DE RUIDO IMAGEN
            Denoised_t1 = medianFilterBorders(Estandarize_t1)
            Denoised_ir = medianFilterBorders(Estandarize_ir)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(Denoised_t1[:,:,25])
            ax1.set_title('T1')
            ax2.imshow(Denoised_ir[:,:,25])
            ax2.set_title('IR')
            st.pyplot(fig)

            # Obtener la información de afine de la imagen original
            affine_t1 = image_T1.affine
            affine_ir = image_IR.affine
            
            # Reconstruir la imagen estandarizada con la información de afine
            reconstructed_image_t1 = nib.Nifti1Image(Denoised_t1.astype(np.float32), affine_t1)
            reconstructed_image_ir = nib.Nifti1Image(Denoised_ir.astype(np.float32), affine_ir)
            
            # Guardar la imagen estandarizada en formato NIfTI
            output_path_t1 = os.path.join("temp", "Preprocess_t1.nii.gz")
            output_path_ir = os.path.join("temp", "Preprocess_ir.nii.gz")
            nib.save(reconstructed_image_t1, output_path_t1)
            nib.save(reconstructed_image_ir, output_path_ir)
            
            # Agregar un enlace para descargar el archivo NIfTI generado
            st.success("Imagen estandarizada guardada correctamente.")
            # Agregar el botón de descarga
            if os.path.exists(output_path_t1):
                with open(output_path_t1, "rb") as file:
                    st.download_button("Descargar archivo", data=file, file_name="Preprocess_t1.nii.gz")

            if os.path.exists(output_path_ir):
                with open(output_path_ir, "rb") as file:
                    st.download_button("Descargar archivo", data=file, file_name="Preprocess_ir.nii.gz")