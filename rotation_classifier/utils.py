import tensorflow as tf

INPUT_DIMS = (100, 100, 3)
CLASSES = ['Left Turn', 'Right Turn', 'No Turn']

# train over imagenet dataset, works well
RESNET_MODEL = tf.keras.applications.ResNet152(input_shape=INPUT_DIMS,
                                               include_top=False,
                                               weights='imagenet')

VGG_MODEL = tf.keras.applications.VGG16(input_shape=INPUT_DIMS,
                                        include_top=False,
                                        weights='imagenet')

BATCH_SIZE = 32
RAND_SEED = 1
PREPROCESSING_DIMS = (100, 1000)
DET_THRESH = 0.5
BUFFER_SIZE = 20
