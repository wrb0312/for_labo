"""
0;95;0cEdit by Keisuke Oyamada @2018/04/19.
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
    batch_size = 10
    n_cls = 10

    print("MNIST example for test")
    print("gpu ID: {}".format(gpu))
    print("model path: {}".format(model_path))
    print("save path: {}".format(save_path))

    G = net.generator(z_dim=args.z_dim, width=7, ch=256, n_cls=n_cls)

    if gpu >= 0:
        chainer.backends.cuda.get_device_from_id(gpu).use()
        G.to_gpu()

    xp = np if gpu < 0 else cupy

    serializers.load_npz(model_path, G)

    z = chainer.Variable(xp.random.randn(batch_size, args.z_dim).astype(xp.float32))
    idx = 0
    plt.figure(figsize=(n_cls, batch_size))

    for i in range(n_cls):
        c = [i]*batch_size
        c = np.eye(n_cls)[c].astype(np.float32)
        c = xp.asarray(c)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x_fake = G(z, c)
            x_fake = chainer.backends.cuda.to_cpu(x_fake.data)

        for j in range(batch_size):
            plt.subplot(n_cls, batch_size, idx+1)
            plt.imshow(x_fake[j, 0], cmap="gray")
            plt.axis("off")
            idx += 1

    plt.tight_layout()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, "result.jpg"), bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    main()
