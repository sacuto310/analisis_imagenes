import SimpleITK as sitk
import numpy as np
import os
import nibabel as nib
from segmentations.kmeans import kmeans_segmentation
from scipy import ndimage

def registro_rigido(name, imagen_movil, imagen_referencia, imagen_seg, output_folder="./temp/"):
    # Cargar las imágenes usando SimpleITK
    imagen_movil_sitk = sitk.ReadImage(imagen_movil)
    imagen_referencia_sitk = sitk.ReadImage(imagen_referencia)
    imagen_seg_sitk = sitk.ReadImage(imagen_seg)

    # Convertir la imagen móvil a tipo de datos float32
    imagen_movil_sitk = sitk.Cast(imagen_movil_sitk, sitk.sitkFloat32)
    imagen_seg_sitk = sitk.Cast(imagen_seg_sitk, sitk.sitkFloat32)
    imagen_referencia_sitk = sitk.ReadImage(imagen_referencia, sitk.sitkFloat32)

    # Crear el objeto de registro rígido
    registro_rigido = sitk.ImageRegistrationMethod()

    # Configurar los parámetros del registro rígido
    registro_rigido.SetMetricAsMeanSquares()
    registro_rigido.SetOptimizerAsRegularStepGradientDescent(learningRate=0.1, minStep=1e-4, numberOfIterations=100)
    registro_rigido.SetInitialTransform(sitk.TranslationTransform(imagen_movil_sitk.GetDimension()))

    # Realizar el registro rígido
    transformada_resultado = registro_rigido.Execute(imagen_referencia_sitk, imagen_movil_sitk)

    # Aplicar la transformación alineada a la imagen móvil completa
    imagen_movil_registrada = sitk.Resample(imagen_seg_sitk, imagen_referencia_sitk, transformada_resultado, sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat64)

    # Obtener la matriz tridimensional de la imagen móvil registrada y reordenar las dimensiones
    matriz_registrada = sitk.GetArrayFromImage(imagen_movil_registrada)
    matriz_registrada = np.transpose(matriz_registrada, (2, 1, 0))

    # Guardar la imagen registrada en formato NIfTI
    output_path = os.path.join(output_folder, name+".nii.gz")
    image_segmented = nib.Nifti1Image(matriz_registrada, affine=np.eye(4))
    nib.save(image_segmented, output_path)
    print(f"Imagen segmentada guardada en {output_path}")

    return matriz_registrada

def remove_brain(output_folder="./temp/", uploads_folder="./uploads/"):
    # Cargar la imagen NIfTI

    nifti_img = nib.load(
        os.path.join(output_folder, "ir_reg.nii.gz")
    )  # Asegúrate de ajustar la ruta y el nombre del archivo

    # Obtener los datos de la imagen
    data = nifti_img.get_fdata()

    # Definir escalas espaciales
    scales = [13.5]  # Escalas para aplicar filtros gaussianos

    # Aplicar filtros gaussianos en diferentes escalas
    filtered_images = []
    for scale in scales:
        # Aplicar filtro gaussiano
        filtered = ndimage.gaussian_filter(data, sigma=scale)
        filtered = kmeans_segmentation(filtered, 10, 4)
        # Crear una nueva imagen nibabel con el cerebro extraído
        brain_extracted_image = nib.Nifti1Image(
            filtered, affine=nifti_img.affine, dtype=np.int16
        )

        # Guardar la imagen con el cerebro extraído en un nuevo archivo
        nib.save(brain_extracted_image, os.path.join(output_folder, "IR_skull.nii.gz"))
        filtered_images.append(filtered)

    # RESTAR UNA IMAGEN

    # Cargar las imágenes
    imagen_original = sitk.ReadImage(
        os.path.join(output_folder, "t1_reg.nii.gz")
    )
    imagen_referencia = sitk.ReadImage(os.path.join(output_folder, "IR_skull.nii.gz"))

    # Modify the metadata of image2 to match image1
    imagen_referencia.SetOrigin(imagen_original.GetOrigin())
    imagen_referencia.SetSpacing(imagen_original.GetSpacing())
    imagen_referencia.SetDirection(imagen_original.GetDirection())

    # Realizar segmentación basada en umbral adaptativo
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0)
    mascara_referencia = otsu_filter.Execute(imagen_referencia)

    # Aplicar la máscara a la imagen original
    imagen_sin_craneo = sitk.Mask(imagen_original, mascara_referencia)

    # Obtener los datos de la imagen sin el cráneo
    # Obtener los datos de la imagen sin el cráneo
    data_sin_craneo = sitk.GetArrayFromImage(imagen_sin_craneo)

    # Obtener los datos de la máscara
    data_mascara = sitk.GetArrayFromImage(mascara_referencia)

    # Crear una máscara booleana para los valores cero dentro del cerebro
    mascara_cero_cerebro = (data_sin_craneo == 0) & (data_mascara != 0)

    # Asignar un valor distinto a los valores cero dentro del cerebro
    valor_distinto = 6
    data_sin_craneo[mascara_cero_cerebro] = valor_distinto

    # Crear una nueva imagen SimpleITK con los datos modificados
    imagen_sin_craneo_modificada = sitk.GetImageFromArray(data_sin_craneo)
    imagen_sin_craneo_modificada.CopyInformation(imagen_sin_craneo)

    # Guardar la imagen sin el cráneo

    sitk.WriteImage(
        imagen_sin_craneo_modificada, os.path.join(output_folder, "FLAIR_skull.nii.gz")
    )

    # ----------------------------------------------------------------------------------
    # Quitar cráneo a FLAIR Original
    # ----------------------------------------------------------------------------------
    # Cargar las imágenes

    imagen_original = sitk.ReadImage(os.path.join(uploads_folder, "FLAIR.nii.gz"))
    imagen_referencia = sitk.ReadImage(os.path.join(output_folder, "IR_skull.nii.gz"))

    imagen_referencia.SetOrigin(imagen_original.GetOrigin())
    imagen_referencia.SetSpacing(imagen_original.GetSpacing())
    imagen_referencia.SetDirection(imagen_original.GetDirection())


    # Realizar segmentación basada en umbral adaptativo
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0)
    mascara_referencia = otsu_filter.Execute(imagen_referencia)

    # Aplicar la máscara a la imagen original
    imagen_sin_craneo = sitk.Mask(imagen_original, mascara_referencia)

    # Guardar la imagen sin el cráneo

    sitk.WriteImage(
        imagen_sin_craneo,
        os.path.join(output_folder, "FLAIR_original_sin_craneo.nii.gz"),
    )

    # ----------------------------------------------------------------------------------
    # Segmentar lesiones
    # ----------------------------------------------------------------------------------

    image = nib.load(os.path.join(output_folder, "FLAIR_skull.nii.gz"))
    image_data = image.get_fdata()
    image_data_flair_without_skull = nib.load(
        os.path.join(output_folder, "FLAIR_original_sin_craneo.nii.gz")
    ).get_fdata()

    image_data_flair_segmented = kmeans_segmentation(image_data_flair_without_skull, 15, 15)

    # Where the values are 3, replace them in the image_data with a value of 3
    image_data_flair_segmented[:,:,:13] = 0
    image_data = np.where(image_data_flair_segmented == 7, 3, image_data)

    affine = image.affine
    # Create a nibabel image object from the image data
    image = nib.Nifti1Image(image_data.astype(np.float32), affine=affine)
    # Save the image as a NIfTI file
    output_path = os.path.join(output_folder, "FLAIR_skull_lesion.nii.gz")
    nib.save(image, output_path)

    return image_data


def volumes(image_path):
    image_flair = nib.load("../uploads/FLAIR.nii.gz")

    image = nib.load("../temp/"+image_path)
    image_data = image.get_fdata()

    unique, counts = np.unique(image_data.astype(np.int32), return_counts=True)
    # print(unique)
    voxel_size = np.abs(image_flair.affine.diagonal()[:3])

    # Create an array of size unique values
    volume = [None] * len(unique)

    # For each unique value, calculate volume
    for i in range(len(unique)):
        count = 0
        volume_in_mm3 = 0

        count = np.count_nonzero(image_data.astype(np.int32) == i)
        volume_in_mm3 = count * np.prod(voxel_size)

        volume[i] = [count, volume_in_mm3]

        # Imprimir el resultado
        # print(f"LABEL #{i} - Cantidad de voxeles: {count} - volumen en mm3: {volume_in_mm3.round(2)}")

    return volume