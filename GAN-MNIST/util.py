import cv2
import scipy.misc
import numpy as np
import os

def OneHot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh


def crop_resize(image_path, resize_shape=(64,64)):
    image = cv2.imread(image_path)
    height, width, channel = image.shape

    if width == height:
        resized_image = cv2.resize(image, resize_shape)
    elif width > height:
        resized_image = cv2.resize(image, (int(width * float(resize_shape[0])//height), resize_shape[1]))
        cropping_length = int( (resized_image.shape[1] - resize_shape[0]) // 2)
        resized_image = resized_image[:,cropping_length:cropping_length+resize_shape[1]]
    else:
        resized_image = cv2.resize(image, (resize_shape[0], int(height * float(resize_shape[1])/width)))
        cropping_length = int( (resized_image.shape[0] - resize_shape[1]) // 2)
        resized_image = resized_image[cropping_length:cropping_length+resize_shape[0], :]

    return resized_image/127.5 - 1

def save_visualization(X, nh_nw, save_path='./vis/sample.jpg'):
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))

    for n,x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = x

    scipy.misc.imsave(save_path, img)

def save_data_set(X, labels, flags, save_path='./raw'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h, w))
    for n, x in enumerate(X):
        label = np.argmax(labels[n])
        img[:,:] = x[:,:,0]
        os_dir = '%s/%d' % (save_path, label)
        if not os.path.exists(os_dir):
            os.mkdir(os_dir)
        scipy.misc.imsave('%s/%d/im%04d.png' % (save_path, label, flags[label]), img)
        flags[label] += 1   # Always be careful that it will change the origin array flags

# verify the bug in JPG to MNIST format
def save_data_set_3C(X, labels, flags, save_path='./raw_3C'):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h, w, 3))
    for n, x in enumerate(X):
        label = np.argmax(labels[n])
        img[:,:,:] = x[:,:,:]
        os_dir = '%s/%d' % (save_path, label)
        if not os.path.exists(os_dir):
            os.mkdir(os_dir)
        scipy.misc.imsave('%s/%d/im%04d.png' % (save_path, label, flags[label]), img)
        flags[label] += 1   # Always be careful that it will change the origin array flags

