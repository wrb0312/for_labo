import chainer
import numpy as np
from tqdm import tqdm
import math
import os
from chainer import serializers
import chainer.functions as F
from reporter import reporter
from chainer import dataset


def update(model, opt, loss, is_unchain=False):
    model.cleargrads()
    loss.backward()
    opt.update()
    if is_unchain:
        loss.unchain_backward()


class Trainer:
    def __init__(self, **kwargs):
        self.G, self.D = kwargs.pop("model")
        self.opt_G, self.opt_D = kwargs.pop("opt")
        self.epochs = kwargs.pop("epochs")
        self.save_path = kwargs.pop("save_path")
        self.iter_train, self.iter_test = kwargs.pop("iterator")
        self.n_cls = kwargs.pop("n_cls")
        self.hight, self.width = kwargs.pop("img_size")
        N, N_test = kwargs.pop("data_size")
        batch_size, batch_size_test = kwargs.pop("batch_size")
        self.z_dim = kwargs.pop("z_dim")
        self.xp = self.G.xp

        self.keys_general = ["epoch"]
        self.keys_train = ["loss_D_train",
                           "loss_G_train"]
        self.keys_test = ["loss_D_test",
                          "loss_G_test"]
        self.report = reporter(self.keys_general, self.keys_train, self.keys_test, N, N_test, self.save_path)
        self.N, self.N_test = math.ceil(N / batch_size), math.ceil(N_test / batch_size_test)

    def __call__(self):
        for epoch in range(1, self.epochs+1):
            self.report.epoch = epoch

            ### train ###
            num_train = 0
            bar = tqdm(desc="Training", total=self.N, leave=False)
            for d in self.iter_train:
                x, c = dataset.concat_examples(d)
                loss_D, loss_G = self.forward(x, c)
                self.report.loss_D_train += float(loss_D * len(x))
                self.report.loss_G_train += float(loss_G * len(x))
                num_train += len(x)

                bar.set_description("epoch: {}, loss_D: {:.6f}, loss_G: {:.6f}".format(
                    epoch,
                    self.report.loss_D_train / num_train,
                    self.report.loss_G_train / num_train),
                    refresh=False)
                bar.update()
            bar.close()

            ### test ###
            num_test = 0
            bar_test = tqdm(desc="Test", total=self.N_test, leave=False)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                for d in self.iter_test:
                    x, c = dataset.concat_examples(d)
                    loss_D, loss_G = self.forward(x, c)
                    self.report.loss_D_test += float(loss_D * len(x))
                    self.report.loss_G_test += float(loss_G * len(x))
                    num_test += len(x)
                    bar_test.update()
            bar_test.close()

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            if not os.path.exists(os.path.join(self.save_path, "models")):
                os.makedirs(os.path.join(self.save_path, "models"))

            self.report(self.keys_general, self.keys_train, self.keys_test)

            self.iter_train.reset()
            self.iter_test.reset()
            self.report.init_log()

            serializers.save_npz(os.path.join(
                self.save_path, "models", "epoch{}.gen".format(epoch)), self.G)
            serializers.save_npz(os.path.join(
                self.save_path, "models", "epoch{}.dis".format(epoch)), self.D)
            serializers.save_npz(os.path.join(
                self.save_path, "models", "epoch{}.opt_gen".format(epoch)), self.opt_G)
            serializers.save_npz(os.path.join(
                self.save_path, "models", "epoch{}.opt_dis".format(epoch)), self.opt_D)

    def forward(self, x, c_real):
        batch_size = len(x)
        x = x * 2 - 1
        x = x.reshape(batch_size, 1, self.hight, self.width)
        x_real = chainer.Variable(self.xp.asarray(x))
        c_real = np.eye(self.n_cls)[c_real]
        c_real = chainer.Variable(self.xp.asarray(c_real).astype(self.xp.float32))
        c_real = F.tile(c_real.reshape(batch_size, self.n_cls, 1, 1), (self.hight, self.width))
        y_real = self.D(x_real, c_real)

        z = chainer.Variable(self.xp.random.randn(batch_size, self.z_dim).astype(self.xp.float32))
        c_fake = np.random.randint(0, self.n_cls, batch_size)
        c_fake = np.eye(self.n_cls)[c_fake]
        c_fake = chainer.Variable(self.xp.asarray(c_fake).astype(self.xp.float32))
        x_fake = self.G(z, c_fake)
        c_fake = F.tile(c_fake.reshape(batch_size, self.n_cls, 1, 1), (self.hight, self.width))
        y_fake = self.D(x_fake, c_fake)

        label_real = chainer.Variable(self.xp.ones((batch_size, 1)).astype(self.xp.int32))
        label_fake = chainer.Variable(self.xp.zeros((batch_size, 1)).astype(self.xp.int32))

        loss_D = F.sigmoid_cross_entropy(y_real, label_real) + F.sigmoid_cross_entropy(y_fake, label_fake)
        loss_G = F.sigmoid_cross_entropy(y_fake, label_real)

        if chainer.config.train:
            update(self.D, self.opt_D, loss_D)
            update(self.G, self.opt_G, loss_G, is_unchain=True)

        return loss_D.data, loss_G.data
