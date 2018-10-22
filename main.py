#!/usr/bin/env python
# Apache 2.0 License: https://developers.google.com/machine-learning/practica/image-classification/
import os

BASE_DIR = "./images/"
IMAGE_DATA_WIDTH = 150
WEIGHT_PATH = os.path.abspath("model_weights.best.hdf5")
MODEL_PATH = os.path.abspath("model.hdf5")
TRAINING_EPOCHS = 15

training_dir = os.path.join(BASE_DIR, 'training')
evaluation_dir = os.path.join(BASE_DIR, 'evaluation')

from tensorflow.keras import layers, Model, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All RGB singals will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
evaluation_datagen = ImageDataGenerator(rescale=1./255)

def simp_cnn_model():
    model = models.Sequential()

    # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
    # the three color channels: R, G, and B
    input_shape = (IMAGE_DATA_WIDTH, IMAGE_DATA_WIDTH, 3)

    # First convolution extracts 16 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    model.add(layers.Conv2D(16, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(2))

    # Second convolution extracts 32 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    model.add(layers.Conv2D(32, 3, activation='relu'))
    model.add(layers.MaxPooling2D(2))

    # Third convolution extracts 64 filters that are 3x3
    # Convolution is followed by max-pooling layer with a 2x2 window
    model.add(layers.Conv2D(64, 3, activation='relu'))
    model.add(layers.MaxPooling2D(2))

    model.add(layers.Flatten())
    # Create a fully connected layer with ReLU activation and 512 hidden units
    model.add(layers.Dense(512, activation='relu'))
    # Prevent overfitting
    model.add(layers.Dropout(0.5))
    # Create output layer with a single node and sigmoid activation
    model.add(layers.Dense(1, activation='sigmoid'))

    # Create model:
    # input = input feature map
    # output = input feature map + stacked convolution/maxpooling layers + fully
    # connected layer + sigmoid output layer

    from tensorflow.keras.optimizers import RMSprop
    #from tensorflow.train import RMSPropOptimizer # Not used due to tensorflow/tensorflow#20999

    #model_optimizer = RMSPropOptimizer(learning_rate=0.001)
    #model_optimizer.lr = 0.001

    model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(lr=0.001),
                metrics=['acc'])

    return model

def pix2code_model():
    model = models.Sequential()

    input_shape = (IMAGE_DATA_WIDTH, IMAGE_DATA_WIDTH, 3)
    model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), padding='valid', activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), padding='valid', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))

    from tensorflow.keras.optimizers import RMSprop
    #from tensorflow.train import RMSPropOptimizer # Not used due to tensorflow/tensorflow#20999

    #model_optimizer = RMSPropOptimizer(learning_rate=0.0001)

    model.compile(loss='binary_crossentropy',
                optimizer=RMSprop(lr=0.0001, clipvalue=1.0),
                metrics=['acc'])

    return model

def train():
    training_ios_dir = os.path.join(training_dir, 'ios')
    training_android_dir = os.path.join(training_dir, 'android')
    evaluation_ios_dir = os.path.join(evaluation_dir, 'ios')
    evaluation_android_dir = os.path.join(evaluation_dir, 'android')

    print('total training ios images:', len(os.listdir(training_ios_dir)))
    print('total training android images:', len(os.listdir(training_android_dir)))
    print('total evaluation ios images:', len(os.listdir(evaluation_ios_dir)))
    print('total evaluation android images:', len(os.listdir(evaluation_android_dir)))

    if os.path.exists(MODEL_PATH):
        model = models.load_model(MODEL_PATH)
    else:
        model = simp_cnn_model()

    model.summary()

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
            training_dir,  # This is the source directory for training images
            target_size=(IMAGE_DATA_WIDTH, IMAGE_DATA_WIDTH),  # All images will be resized to 150x150
            shuffle=True, seed=7,
            batch_size=50,
            color_mode='rgb',
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary',
            subset="training")

    validation_generator = validation_datagen.flow_from_directory(
            training_dir,
            target_size=(IMAGE_DATA_WIDTH, IMAGE_DATA_WIDTH),
            shuffle=True, seed=7,
            batch_size=10,
            color_mode='rgb',
            class_mode='binary',
            subset="validation")

    from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

    checkpoint = ModelCheckpoint(WEIGHT_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    #reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=1, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=1e-8)
    #early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=20)

    callbacks_list = [checkpoint, ]#, early, reduceLROnPlat]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=30,  # 2000 images = batch_size * steps
        epochs=TRAINING_EPOCHS,
        validation_data=validation_generator,
        validation_steps=10,  # 1000 images = batch_size * steps
        callbacks=callbacks_list,
        workers=1,
        verbose=1)

    model.load_weights(WEIGHT_PATH)
    model.save(MODEL_PATH)

    # Retrieve a list of accuracy results on training and test data
    # sets for each training epoch
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    # Plot training and validation accuracy per epoch
    print('  acc\tloss\tval_a\tval_l')
    for values in zip(acc, loss, val_acc, val_loss):
        print('{:.04f}\t{:.04f}\t{:.04f}\t{:.04f}'.format(*values))

    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')

    plt.figure()

    # Plot training and validation loss per epoch
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')

def evaluate():
    model = models.load_model(MODEL_PATH)
    evaluation_generator = evaluation_datagen.flow_from_directory(
            evaluation_dir,
            target_size=(IMAGE_DATA_WIDTH, IMAGE_DATA_WIDTH),
            batch_size=10,
            color_mode='rgb',
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='binary',
        )

    result = model.evaluate_generator(evaluation_generator)
    predict_res = model.predict_generator(evaluation_generator)
    class_indices = {v: k for k, v in evaluation_generator.class_indices.items()}
    class_indices_list = list(class_indices.values())
    total = {v:0 for v in class_indices_list}
    correctness = dict(total)
    correctness_random = dict(total)
    import random
    lets_random = [random.choice(class_indices_list) for i in range(len(evaluation_generator.filenames))]
    for name, value, random_class in zip(evaluation_generator.filenames, predict_res, lets_random):
        predict_class = class_indices[0] if value[0] <= 0.5 else class_indices[1]
        actual_class = name.split('/')[0]
        total[actual_class] = total[actual_class] + 1
        if predict_class == actual_class:
            correctness[actual_class] = correctness[actual_class] + 1
        if random_class == actual_class:
            correctness_random[actual_class] = correctness_random[actual_class] + 1
        # print('{:24.24}\t{:.2f}\t{:<8}'.format(name, value[0], predict_class))
    for key in total.keys():
        print('{}: {}/{} ({:.1%}, rand: {:.1%})'.format(key, correctness[key], total[key], correctness[key] / total[key], correctness_random[key] / total[key]))
    for (name, value) in zip(model.metrics_names, result):
        print('{} = {}'.format(name, value))

def test(img_src):
    pass