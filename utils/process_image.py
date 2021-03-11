import tensorflow as tf

IMG_SIZE = 224


def process_image(img_path):
    '''
    Takes an image file path turns the image into a Tensor
    '''

    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image
