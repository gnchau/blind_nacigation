import tensorflow as tf
from utils import VGG16_MODEL, INPUT_DIMS, BATCH_SIZE, RAND_SEED
from os import system


def create_shift_classifier(arch=None,
                            optimizer=None,
                            loss=None,
                            metrics=None):
    """
    Creates a sidewalk shift classification model. TODO: consolidate shift and rotation factories into one function
    """
    # default vgg since resnet yields inconsistent results
    if not arch:
        arch = 'VGG16'
    else:
        arch = 'RESNET'

    if not optimizer:
        optimizer = tf.keras.optimizers.Adam()
    if not loss:
        loss = 'categorical_crossentropy'
    if not metrics:
        metrics = ['acc']

    global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
    output_layer = tf.keras.layers.Dense(3, activation='softmax')

    if arch == 'RESNET':
        model = tf.keras.Sequential([
            VGG16_MODEL,
            global_avg_layer,
            output_layer
        ])

    elif arch == 'VGG16':
        model = tf.keras.Sequential([
            VGG16_MODEL,
            global_avg_layer,
            output_layer
        ])
    else:
        # only two archs supported
        raise Exception('Must specify a valid architecture.')

    model.compile(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics
    )

    sidewalk_path = '../data/cityscapes/shift_classifier'
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(sidewalk_path,
                                                                   labels='inferred',
                                                                   batch_size=BATCH_SIZE,
                                                                   validation_split=0.1,
                                                                   subset='training',
                                                                   seed=RAND_SEED,
                                                                   image_size=(INPUT_DIMS[0], INPUT_DIMS[1]),
                                                                   label_mode='categorical')

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(sidewalk_path,
                                                                        labels='inferred',
                                                                        batch_size=BATCH_SIZE,
                                                                        validation_split=0.1,
                                                                        subset='validation',
                                                                        seed=RAND_SEED,
                                                                        image_size=(INPUT_DIMS[0], INPUT_DIMS[1]),
                                                                        label_mode='categorical')

    transformation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation((-0.02, 0.02))])

    train_ds = train_ds.map(lambda x, y: (transformation(x), y))
    train_ds = train_ds.cache()
    model.fit(train_ds, validation_data=validation_ds, epochs=8)

    system('mkdir -p ../models')
    out_path = f'../models/sidewalk_classification_model_shift_{arch}.h5'
    model.save(out_path)
