"""
Edit by Keisuke Oyamada @2018/04/19.
For training bachelor.
"""

import chainer
import argparse
import net
from chainer import optimizers
import trainer
from collections import OrderedDict
from chainer.datasets import tuple_dataset
import math


def main():
    parser = argparse.ArgumentParser(description="MNIST example")
    parser.add_argument("--epochs", "-e", help="number of epochs", default=20, type=int)
    parser.add_argument("--gpu", "-g", help="GPU ID (-1 : CPU)", default=-1, type=int)
    parser.add_argument("--batch_size", "-b", help="batch size", default=100, type=int)
    parser.add_argument("--save_path", "-s", help="data save path", required=True, type=str)

    args = parser.parse_args()

    epochs = args.epochs
    gpu = args.gpu
    save_path = args.save_path
    batch_size = args.batch_size
    batch_size_test = batch_size*4

    print("MNIST example")
    for i in OrderedDict(args.__dict__):
        print("{}: {}".format(i, getattr(args, i)))

    mnist_train, mnist_test = chainer.datasets.get_mnist()

    N = len(mnist_train)
    N_test = len(mnist_test)

    print("train data size: {}".format(N))
    print("test data size: {}".format(N_test))

    model = net.sampleNet()

    if gpu >= 0:
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    opt = optimizers.Adam(1e-3)
    opt.setup(model)

    iter_train = chainer.iterators.SerialIterator(mnist_train, batch_size, repeat=False, shuffle=True)
    iter_test = chainer.iterators.SerialIterator(mnist_test, batch_size_test, repeat=False, shuffle=True)

    Trainer = trainer.Trainer(**{
        "model": model,
        "opt": opt,
        "epochs": epochs,
        "save_path": save_path,
        "iterator": (iter_train, iter_test),
        "data_size": (N, N_test),
        "batch_size": (batch_size, batch_size_test),
    })

    Trainer()


if __name__ == "__main__":
    main()
