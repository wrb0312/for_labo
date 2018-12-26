"""
Edit by Keisuke Oyamada @2018/04/19.
For training bachelor.
"""

import numpy as np
import chainer
import cupy
import argparse
import net
from chainer import serializers
from chainer import functions as F
import os
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="MNIST example for test")
    parser.add_argument("--gpu", "-g", help="GPU ID (-1 : CPU)", default=-1, type=int)
    parser.add_argument("--model_path", "-m", help="model path", required=True, type=str)
    parser.add_argument("--save_path", "-s", help="data save path", required=True, type=str)

    args = parser.parse_args()

    gpu = args.gpu
    model_path = args.model_path
    save_path = args.save_path
    batch_size = 20

    print("MNIST example for test")
    print("gpu ID: {}".format(gpu))
    print("model path: {}".format(model_path))
    print("save path: {}".format(save_path))

    _, mnist_test = chainer.datasets.get_mnist()
    x_test, y_test = mnist_test._datasets

    N_test = len(x_test)
    print("test data size: {}".format(N_test))

    model = net.sampleNet()

    if gpu >= 0:
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    xp = np if gpu < 0 else cupy

    serializers.load_npz(model_path, model)

    perm_test = np.random.permutation(N_test)
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x_test_v = xp.asarray(x_test[perm_test[:batch_size]])
        y_test_v = y_test[perm_test[:batch_size]]

        y_test_h = model(x_test_v)

        y_test_h = chainer.backends.cuda.to_cpu(F.softmax(y_test_h).data)
        x_test_v = chainer.backends.cuda.to_cpu(x_test_v)

    c = 0
    plt.figure(figsize=(10, 10))
    for i in range(batch_size):
        plt.subplot(5, 4, i+1)
        plt.imshow(x_test_v[c].reshape(28, 28))
        plt.axis("off")
        plt.title("pred: {}, truth: {}".format(np.argmax(y_test_h[c]), y_test_v[c]))
        c += 1

    plt.tight_layout()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "result.png"))

if __name__ == "__main__":
    main()
