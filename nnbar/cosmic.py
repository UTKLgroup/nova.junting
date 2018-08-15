from rootalias import *
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib


def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
       ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model = tf.keras.models.Sequential([
    #     keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
    #     keras.layers.Dropout(0.2),
    #     keras.layers.Dense(10, activation=tf.nn.softmax)
    # ])
    # model.compile(optimizer=tf.keras.optimizers.Adam(),
    #               loss=tf.keras.losses.sparse_categorical_crossentropy,
    #               metrics=['accuracy'])

    return model


def test_tensorflow():
    # print(tf.__version__)
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print('train_images.shape = {}'.format(train_images.shape))
    print('len(train_labels) = {}'.format(len(train_labels)))
    print('train_labels = {}'.format(train_labels))

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.
    test_images = test_images / 255.

    # plt.figure()
    # plt.imshow(train_images[0])
    # plt.colorbar()
    # plt.gca().grid(False)

    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid('off')
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])

    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(28, 28)),
    #     keras.layers.Dense(128, activation=tf.nn.relu),
    #     keras.layers.Dense(10, activation=tf.nn.softmax)
    #    ])
    # model.compile(optimizer=tf.train.AdamOptimizer(),
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    model = create_model()
    print('model.summary() = {}'.format(model.summary()))

    # checkpoint_path = "tmp/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
    #                                                  save_weights_only=True,
    #                                                  verbose=1)
    # model.fit(train_images, train_labels, epochs=5,
    #           validation_data = (test_images,test_labels),
    #           callbacks = [cp_callback])

    # model.fit(train_images, train_labels, epochs=5)
    # model.save_weights('./tmp/my_checkpoint')

    model.fit(train_images, train_labels, epochs=5)
    model.save('./tmp/my_model.h5')

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

    predictions = model.predict(test_images)
    print('predictions[0] = {}'.format(predictions[0]))
    print('np.argmax(predictions[0]) = {}'.format(np.argmax(predictions[0])))
    print('test_labels[0] = {}'.format(test_labels[0]))

    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid('off')
    #     plt.imshow(test_images[i], cmap=plt.cm.binary)
    #     predicted_label = np.argmax(predictions[i])
    #     true_label = test_labels[i]
    #     if predicted_label == true_label:
    #         color = 'green'
    #     else:
    #         color = 'red'
    #     plt.xlabel("{} ({})".format(class_names[predicted_label], class_names[true_label]),
    #                color=color)

    img = test_images[0]
    print(img.shape)
    img = (np.expand_dims(img,0))
    print(img.shape)
    predictions = model.predict(img)
    print(predictions)
    prediction = predictions[0]
    print('np.argmax(prediction) = {}'.format(np.argmax(prediction)))

    # plt.show()


def test_load_model():
    # checkpoint_path = "tmp/cp.ckpt"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # checkpoints = pathlib.Path(checkpoint_dir).glob("*.index")
    # print('glob')
    # print('checkpoints = {}'.format(checkpoints))
    # checkpoints = sorted(checkpoints, key=lambda cp:cp.stat().st_mtime)
    # print('sort')
    # print('checkpoints = {}'.format(checkpoints))
    # checkpoints = [cp.with_suffix('') for cp in checkpoints]
    # print('suffix')
    # print('checkpoints = {}'.format(checkpoints))
    # latest = str(checkpoints[-1])
    # print('latest = {}'.format(latest))

    # model = create_model()
    # # model.load_weights(latest)
    # model.load_weights('./tmp/my_checkpoint')

    model = keras.models.load_model('./tmp/my_model.h5')
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_images = train_images / 255.
    test_images = test_images / 255.
    loss, acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))


def test_load_data():
    # data = np.random.random((1000, 32))
    # labels = np.random.random((1000, 10))

    # val_data = np.random.random((100, 32))
    # val_labels = np.random.random((100, 10))

    # print('data = {}'.format(data))
    # print('labels = {}'.format(labels))

    # model = keras.Sequential()
    # model.add(keras.layers.Dense(64, activation='relu'))
    # model.add(keras.layers.Dense(64, activation='relu'))
    # model.add(keras.layers.Dense(10, activation='softmax'))
    # model.compile(optimizer=tf.train.AdamOptimizer(0.001),
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.fit(data, labels, epochs=10, batch_size=32,
    #           validation_data=(val_data, val_labels))

    # dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    # # dataset = tf.data.Dataset.from_tensor_slices((tf.cast(data, tf.float32), tf.cast(labels, tf.float32)))
    # dataset = dataset.batch(32).repeat()
    # val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    # val_dataset = val_dataset.batch(32).repeat()
    # model.fit(dataset, epochs=10, steps_per_epoch=30)
    # model.fit(dataset, epochs=10, steps_per_epoch=30,
    #           validation_data=val_dataset,
    #           validation_steps=3)

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # print('train_images[0] = {}'.format(train_images[0]))
    train_images = train_images / 255.
    test_images = test_images / 255.

    print('train_images.shape = {}'.format(train_images.shape))
    print('train_labels.shape = {}'.format(train_labels.shape))


def load_train_data(filename):
    tf = TFile(filename)
    images = []
    for event in tf.Get('neutronoscana/fSliceTree'):
        cells = event.cells
        planes = event.planes
        views = event.views

        image = np.zeros((768, 896))
        for i, cell in enumerate(cells):
            view = views[i]
            plane = planes[i]
            cell = cell
            if view == 1:
                cell += 384
            image[cell][plane] = 1.

        # plt.figure(figsize=(int(896 / 60), int(768 / 60)))
        # plt.imshow(img)
        # plt.colorbar()
        # plt.gca().grid(False)
        # plt.show()
        # break
        images.append(image)

    return np.array(images)


def train():
    noises = load_train_data('data/train.noise.hist.root')
    noise_labels = np.full((1, len(noises)), 1)[0]
    signals = load_train_data('data/train.signal.hist.root')
    signals_labels = np.full((1, len(signals)), 1)[0]
    # print('signals_labels = {}'.format(signals_labels))
    # print('noises.shape = {}'.format(noises.shape))
    print('noise_labels.shape = {}'.format(noise_labels.shape))
    print('signals_labels.shape = {}'.format(signals_labels.shape))

    # plt.figure(figsize=(int(896 / 60), int(768 / 60)))
    # plt.imshow(noises[0])
    # plt.colorbar()
    # plt.gca().grid(False)
    # plt.show()


train()
# test_load_data()
# load_train_data('data/train.noise.hist.root')
# load_train_data('data/train.signal.hist.root')
# test_tensorflow()
# test_load_model()
