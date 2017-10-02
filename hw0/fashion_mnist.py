import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tensorflow.contrib import keras
import scipy
training_data_path='./data/train-images-idx3-ubyte'
training_label_path='./data/train-labels-idx1-ubyte'
testing_data_path='./data/t10k-images-idx3-ubyte'

batch_size=512
epoch = 15000
lr=0.001

def to_onehot(tensor,n_class):
    assert(tensor.ndim == 1)
    assert(tensor.dtype == np.int8)
    one_hot = np.zeros([tensor.shape[0],n_class],dtype=float)
    for i in range(tensor.shape[0]):
        one_hot[i, tensor[i]] = 1.
    return one_hot

def get_data_and_labels(images_filename, labels_filename):
    print("Opening files ...")
    images_file = open(images_filename, "rb")
    labels_file = open(labels_filename, "rb")

    try:
        print("Reading files ...")
        images_file.read(4)
        num_of_items = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_rows = int.from_bytes(images_file.read(4), byteorder="big")
        num_of_colums = int.from_bytes(images_file.read(4), byteorder="big")
        labels_file.read(8)

        num_of_image_values = num_of_rows * num_of_colums
        data = np.zeros([num_of_items,num_of_image_values],dtype=float)
        labels = np.zeros([num_of_items],dtype=int)
        for item in range(num_of_items//batch_size*batch_size):
            data[item,:]=np.fromfile(images_file,dtype=np.uint8,count=784)

        labels=np.fromfile(labels_file,dtype=np.int8,count=num_of_items)

        data=data.reshape([-1,28,28,1])
        return data, to_onehot(labels,10)
    finally:
        images_file.close()
        labels_file.close()
        print("Files closed.")

training_data, labels = get_data_and_labels(training_data_path,training_label_path)
testing_data, _ = get_data_and_labels(testing_data_path,training_label_path)

feed_image=tf.placeholder(tf.float32, [None, 28, 28, 1])
feed_label=tf.placeholder(tf.float32, [None, 10])

def model(feed_image):
    image = feed_image
    conv1=tf.layers.conv2d(image, 64, [3, 3], activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], 1)
    conv2 = tf.layers.conv2d(pool1,64, [3, 3], activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, [2, 2], 1)

    flatten=tf.reshape(pool2,[-1,64*22*22])
    dense1=tf.layers.dense(flatten, 1024, activation=tf.nn.relu)
    dense2 = tf.layers.dense(dense1, 512, activation=tf.nn.relu)
    dense3 = tf.layers.dense(dense2, 512, activation=tf.nn.relu)
    pred=tf.layers.dense(dense3, 10)

    return pred

def get_batch(data,labels):
    n_data=data.shape[0]
    n_select=random.randint(0, n_data-batch_size)
    batch_images = data[n_select:n_select+batch_size,:]
    batch_labels = labels[n_select:n_select+batch_size,:]
    return batch_images, batch_labels

def split_val(training_data,labels,ratio=0.1):
    n_train=int(training_data.shape[0]*(1-ratio))
    train_data=training_data[:n_train]
    train_labels=labels[:n_train]
    val_data=training_data[n_train:]
    val_labels=labels[n_train:]

    return train_data,train_labels,val_data,val_labels


train_data, train_labels, val_data, val_labels = split_val(training_data,labels,0.1)

pred = model(feed_image)
loss = tf.losses.softmax_cross_entropy(onehot_labels=feed_label,logits=pred)
optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(feed_label,1),tf.arg_max(pred,1)), tf.float32))

image_generator=keras.preprocessing.image.ImageDataGenerator(rotation_range=10.,
                                                             width_shift_range=0.1,
                                                             height_shift_range=0.1,
                                                             shear_range=0.,
                                                             zoom_range=0.1,
                                                             horizontal_flip=True)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i,(batch_images, batch_labels) in enumerate(image_generator.flow(x=train_data, y=train_labels,batch_size=batch_size)):
        sess.run(train_op, feed_dict={feed_image: batch_images,
                                      feed_label: batch_labels})

        if (i+1) % 10 == 0:
            print(i, sess.run([loss,accuracy],feed_dict={feed_image: batch_images,
                                                    feed_label: batch_labels}))
            val_batch_ims, val_batch_labels = get_batch(val_data,val_labels)
            print('val_accuracy: ', sess.run([accuracy],feed_dict={feed_image: val_batch_ims,
                                                    feed_label: val_batch_labels}))

        if i > epoch:
            break
    f=open('result.csv',mode='w+')
    f.write('id,label\n')
    for i in range(testing_data.shape[0]):
        test_sample=np.expand_dims(testing_data[i],axis=0)
        test_predict=sess.run(pred,feed_dict={feed_image: test_sample})
        f.write(str(i)+',')
        f.write(str(np.argmax(test_predict,axis=1)[0]))
        f.write('\n')
    f.close()