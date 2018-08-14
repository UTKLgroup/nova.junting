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


# test_tensorflow()
test_load_model()
