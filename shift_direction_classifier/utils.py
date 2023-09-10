import tensorflow as tf

INPUT_DIMS = (100, 100, 3)
VGG16_MODEL = tf.keras.applications.VGG16(input_shape=INPUT_DIMS,
                                          include_top=False,
                                          weights='imagenet')

RESNET_MODEL = tf.keras.applications.ResNet152(input_shape=INPUT_DIMS,
                                               include_top=False,
                                               weights='imagenet')


BATCH_SIZE = 32
CLASSES = ['Left shift from sidewalk',
           'Middle of sidewalk',
           'Right shift of sidewalk']
THRESH = 0.8

PREPROCESSING_DIMS = (100, 100)
BUFFER_SIZE = 50
DET_THRESH = 0.5

RAND_SEED = 1
