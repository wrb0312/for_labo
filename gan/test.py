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
    parser.add_argument("--z_dim", "-z", help="z dimention size", default=128, type=int)

    args = parser.parse_args()

    gpu = args.gpu
    model_path = args.model_path
    save_path = args.save_path
    batch_size = 8

    print("MNIST example for test")
    print("gpu ID: {}".format(gpu))
    print("model path: {}".format(model_path))
    print("save path: {}".format(save_path))

    G = net.generator(z_dim=args.z_dim, width=7, ch=256)

    if gpu >= 0:
        chainer.backends.cuda.get_device_from_id(gpu).use()
        G.to_gpu()

    xp = np if gpu < 0 else cupy

    serializers.load_npz(model_path, G)

    with chainer.using_config('train', False), chainer.no_backprop_mode():
        z = chainer.Variable(xp.random.random((batch_size*batch_size, args.z_dim)).astype(xp.float32))
        x_fake = G(z)
        x_fake = chainer.backends.cuda.to_cpu(x_fake.data)

    plt.figure(figsize=(10, 10))
    for idx, i in enumerate(x_fake):
        plt.subplot(batch_size, batch_size, idx+1)
        plt.imshow(i.reshape(28, 28), cmap="gray")
        plt.axis("off")

    plt.tight_layout()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "result.jpg"), bbox_inches='tight', pad_inches=0)
    plt.close()

    with chainer.using_config('train', False), chainer.no_backprop_mode():
        z = np.random.random((2, args.z_dim)).astype(np.float32)
        z_v = chainer.Variable(xp.asarray(z))
        x_fake = G(z_v)
        x_fake = chainer.backends.cuda.to_cpu(x_fake.data)

    label = ["A", "B"]
    plt.figure(figsize=(2, 2), dpi=200)
    for idx, i in enumerate(x_fake):
        plt.subplot(1, 2, idx+1)
        plt.imshow(i.reshape(28, 28), cmap="gray")
        plt.title(label[idx])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "AB.jpg"), bbox_inches='tight', pad_inches=0)
    plt.close()

    z_new = []
    dist = (z[1] - z[0])/16
    for i in range(16):
        z_new.append(z[0]+(dist*i))
    z_new = xp.asarray(np.array(z_new)).astype(xp.float32)

    with chainer.using_config('train', False), chainer.no_backprop_mode():
        z_v = chainer.Variable(z_new)
        x_fake = G(z_v)
        x_fake = chainer.backends.cuda.to_cpu(x_fake.data)

    plt.figure(figsize=(16, 2), dpi=100)
    for idx, i in enumerate(x_fake):
        plt.subplot(1, 16, idx+1)
        plt.imshow(i.reshape(28, 28), cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "latent_A2B.jpg"), bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    main()
