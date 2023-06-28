import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
from segmentations.kmeans import kmeans_segmentation


def segmentar_process ():
    st.title("Segmentate")

    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    with st.container():

        T1_image = './temp/Preprocess_t1.nii.gz'
        IR_image = './temp/Preprocess_ir.nii.gz'

        # input_files = st.file_uploader('Ingresa las imágenes del paciente', accept_multiple_files=True)

        
        if os.path.exists(T1_image) and os.path.exists(IR_image):
            st.title("Kmeans")
            image_T1 = nib.load(T1_image)
            image_T1_data = image_T1.get_fdata()
            image_IR = nib.load(IR_image)
            image_IR_data = image_IR.get_fdata()
            
            ### SEGMENTAR IMAGEN
            st.title("T1")
            number_of_k_t1 = st.text_input("K # (T1):", key="k_t1")
            iterations_t1 = st.text_input("Iterations (T1):", key="iter_t1")

            st.title("IR")
            number_of_k_ir = st.text_input("K # (IR):", key="k_ir")
            iterations_ir = st.text_input("Iterations (IR):", key="iter_ir")

            buttons_kmeans= st.button("Segmentate")

            if buttons_kmeans:
                # Realizar la segmentación utilizando K-means
                segmentation_t1 = kmeans_segmentation(image_T1_data,int(iterations_t1), int(number_of_k_t1))
                segmentation_ir = kmeans_segmentation(image_IR_data,int(iterations_ir), int(number_of_k_ir))
                
                # Mostrar la imagen segmentada
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.imshow(segmentation_t1[:,:,25])
                ax1.set_title('T1')
                ax2.imshow(segmentation_ir[:,:,25])
                ax2.set_title('IR')
                st.pyplot(fig)
                
                # Obtener la información de afine de la imagen original
                affine_t1 = image_T1.affine
                affine_ir = image_IR.affine
                
                # Reconstruir la imagen estandarizada con la información de afine
                reconstructed_image_t1 = nib.Nifti1Image(segmentation_t1.astype(np.float32), affine_t1)
                reconstructed_image_ir = nib.Nifti1Image(segmentation_ir.astype(np.float32), affine_ir)
                
                # Guardar la imagen estandarizada en formato NIfTI
                output_path_t1 = os.path.join("temp", "Segmentation_t1.nii.gz")
                output_path_ir = os.path.join("temp", "Segmentation_ir.nii.gz")
                nib.save(reconstructed_image_t1, output_path_t1)
                nib.save(reconstructed_image_ir, output_path_ir)
                
                # Agregar un enlace para descargar el archivo NIfTI generado
                st.success("Imagen estandarizada guardada correctamente.")

                # Agregar el botón de descarga
                if os.path.exists(output_path_t1):
                    with open(output_path_t1, "rb") as file:
                        st.download_button("Descargar archivo t1", data=file, file_name="Segmentation_t1.nii.gz")

                if os.path.exists(output_path_ir):
                    with open(output_path_ir, "rb") as file:
                        st.download_button("Descargar archivo ir", data=file, file_name="Segmentation_ir.nii.gz")
        else:
            st.success("No se preprocesaron la imagenes")