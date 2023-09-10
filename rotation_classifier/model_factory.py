import tensorflow as tf
from os import system
from utils import INPUT_DIMS, RESNET_MODEL, VGG_MODEL, BATCH_SIZE, RAND_SEED

# curr implementation only classifies left, right, NA. TODO: angle classifier


def create_rotation_classifier(arch=None,
                               optimizer=None,
                               loss=None,
                               metrics=None):
    # default resnet since vgg yields inconsistent results
    if not arch:
        arch = 'RESNET'
    else: 
        arch = 'VGG16'

    if not optimizer:
        optimizer = tf.keras.optimizers.Adam()
    if not loss:
        loss = 'categorical_crossentropy'
    if not metrics:
        metrics = ['acc']

    global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()
    output_layer = tf.keras.layers.Dense(3, activation='softmax')
    
    # use resnet for rotation classification, vgg inconsistent
    if arch == 'RESNET':
        model = tf.keras.Sequential([
          RESNET_MODEL,
          global_avg_layer,
          output_layer
        ])
    elif arch == 'VGG16':
        model = tf.keras.Sequential([
          VGG_MODEL,
          global_avg_layer,
          output_layer
        ])
    else:
        raise Exception('Must specify a valid architecture.')
    
    model.compile(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics
    )
    
    sidewalk_path = '../data/cityscapes/rotation_classifier'
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(sidewalk_path,
                                                                   labels='inferred',
                                                                   batch_size=BATCH_SIZE,
                                                                   validation_split=0.2,
                                                                   subset="training",
                                                                   seed=RAND_SEED,
                                                                   image_size=(INPUT_DIMS[0], INPUT_DIMS[1]),
                                                                   label_mode="categorical")
    
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(sidewalk_path,
                                                                        labels='inferred',
                                                                        batch_size=BATCH_SIZE,
                                                                        validation_split=0.2,
                                                                        subset="training",
                                                                        seed=RAND_SEED,
                                                                        image_size=(INPUT_DIMS[0], INPUT_DIMS[1]),
                                                                        label_mode="categorical")
    
    train_ds = train_ds.shuffle(buffer_size=1000)
    train_ds = train_ds.cache()
    model.fit(train_ds, validation_data=validation_ds, epochs=4)

    system('mkdir -p ../models')
    out_path = f'../models/rotation_classification_model_{arch}.h5'
    model.save(out_path)
