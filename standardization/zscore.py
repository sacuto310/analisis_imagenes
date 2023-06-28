def zscore_image(image_data):
    mean_value = image_data[image_data > 10].mean()
    standard_deviation_value = image_data[image_data > 10].std()


    image_data_rescaled = (image_data - mean_value) / (standard_deviation_value)

    return image_data_rescaled