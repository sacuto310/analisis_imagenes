import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os
from image_registration.rigid_registration import registro_rigido
from image_registration.rigid_registration import remove_brain
from image_registration.rigid_registration import volumes

def registro_process ():
    st.title("Registro")

    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    with st.container():

        T1_image = './temp/Segmentation_t1.nii.gz'
        IR_image = './temp/Segmentation_ir.nii.gz'

        Flair_original = './uploads/FLAIR.nii.gz'
        T1_original = './uploads/T1.nii.gz'
        IR_original = './uploads/IR.nii.gz'

        # input_files = st.file_uploader('Ingresa las imágenes del paciente', accept_multiple_files=True)

        
        if os.path.exists(T1_image) and os.path.exists(IR_image) and os.path.exists(Flair_original) and os.path.exists(T1_original) and os.path.exists(IR_original):

            volumes = st.button("Calcular volumenes")

            if volumes : 
            
                ##Imagenes segmentadas
                image_T1 = nib.load(T1_image)
                image_T1_data = image_T1.get_fdata()
                image_IR = nib.load(IR_image)
                image_IR_data = image_IR.get_fdata()
                ##Imagenes originales
                image_T1_original = nib.load(T1_original)
                image_T1_data_original = image_T1_original.get_fdata()
                image_IR_original = nib.load(IR_original)
                image_IR_data_original = image_IR_original.get_fdata()
                image_FLAIR_original = nib.load(Flair_original)
                image_FLAIR_data_original = image_FLAIR_original.get_fdata()
                
                ### Registrar IMAGEN

                # Realizar la segmentación utilizando K-means
                registro_t1 = registro_rigido("t1_reg", T1_original, Flair_original, T1_image)
                registro_ir = registro_rigido("ir_reg", IR_original, Flair_original, IR_image)
                Image_final = remove_brain()
                
                # Mostrar la imagen segmentada
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
                ax1.imshow(registro_t1[:,:,25])
                ax1.set_title('T1')
                ax2.imshow(registro_ir[:,:,25])
                ax2.set_title('IR')
                ax3.imshow(Image_final[:,:,25])
                ax3.set_title('FINAL')
                st.pyplot(fig)


                result= np.where(Image_final == 4 , 1 , 0)
                unique,counts=np.unique(Image_final, return_counts=True)
                count= np.count_nonzero(Image_final.astype(np.int32) == 2)
                st.success("Materia Gris: "+ str(counts[1]))
                st.success("Materia Blanca: " + str(counts[2]))
                st.success("Lesiones: " + str(counts[3]))
                st.success("Liquido cefalorraquideo + fondo: " + str(counts[0]))



        else:
            st.success("No se preprocesaron la imagenes")