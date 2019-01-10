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
        N, N_test = kwargs.pop("data_size")
        batch_size, batch_size_test = kwargs.pop("batch_size")
        self.z_dim = kwargs.pop("z_dim")
        self.xp = self.G.xp

        self.keys_general = ["epoch"]
        self.keys_train = ["loss_D_train",
                           "loss_G_train",
                           "loss_REC_train",
                            ]
        self.keys_test = ["loss_D_test",
                          "loss_G_test",
                          "loss_REC_test",
                              ]
        self.report = reporter(self.keys_general, self.keys_train, self.keys_test, N, N_test, self.save_path)
        self.N, self.N_test = math.ceil(N / batch_size), math.ceil(N_test / batch_size_test)

    def __call__(self):
        for epoch in range(1, self.epochs+1):
            self.report.epoch = epoch

            if epoch == 100:
                self.opt_G.alpha = 2e-5
                self.opt_D.alpha = 2e-5
            elif epoch == 300:
                self.opt_G.alpha = 2e-6
                self.opt_D.alpha = 2e-6

            ### train ###
            num_train = 0
            bar = tqdm(desc="Training", total=self.N, leave=False)
            for d in self.iter_train:
                x, _ = dataset.concat_examples(d)
                loss_D, loss_G, loss_REC = self.forward(x)
                self.report.loss_D_train += float(loss_D * len(x))
                self.report.loss_G_train += float(loss_G * len(x))
                self.report.loss_REC_train += float(loss_REC * len(x))
                num_train += len(x)

                bar.set_description("epoch: {}, loss_D: {:.6f}, loss_G: {:.6f}, loss_REC: {:.6f}".format(
                    epoch,
                    self.report.loss_D_train / num_train,
                    self.report.loss_G_train / num_train,
                    self.report.loss_REC_train / num_train,
                    ),
                    refresh=False)
                bar.update()
            bar.close()

            ### test ###
            num_test = 0
            bar_test = tqdm(desc="Test", total=self.N_test, leave=False)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                for d in self.iter_test:
                    x, _ = dataset.concat_examples(d)
                    loss_D, loss_G, loss_REC = self.forward(x)
                    self.report.loss_D_test += float(loss_D * len(x))
                    self.report.loss_G_test += float(loss_G * len(x))
                    self.report.loss_REC_test += float(loss_REC * len(x))
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
            # serializers.save_npz(os.path.join(
            #     self.save_path, "models", "epoch{}.dis".format(epoch)), self.D)
            # serializers.save_npz(os.path.join(
            #     self.save_path, "models", "epoch{}.opt_gen".format(epoch)), self.opt_G)
            # serializers.save_npz(os.path.join(
            #     self.save_path, "models", "epoch{}.opt_dis".format(epoch)), self.opt_D)

    def forward(self, x):
        batch_size = len(x)
        # x = x * 2 - 1
        x = x.reshape(batch_size, 1, 28, 28)
        x_real = chainer.Variable(self.xp.asarray(x))
        z_real = chainer.Variable(self.xp.asarray(np.random.normal(0, 1, (batch_size, 2))).astype(self.xp.float32))
        y_real = self.D(z_real)

        _, z_fake = self.G(x_real)
        y_fake = self.D(z_fake)

        label_real = chainer.Variable(self.xp.ones(batch_size).astype(self.xp.int32))
        label_fake = chainer.Variable(self.xp.zeros(batch_size).astype(self.xp.int32))

        loss_D = F.softmax_cross_entropy(y_real, label_real) + F.softmax_cross_entropy(y_fake, label_fake)

        if chainer.config.train:
            update(self.D, self.opt_D, loss_D*6, True)

        x_fake, _ = self.G(x_real)
        loss_REC = 0.5 * F.sum(F.absolute(x_real - x_fake)) / batch_size

        if chainer.config.train:
            update(self.G, self.opt_G, loss_REC, is_unchain=True)

        _, z_fake = self.G(x_real)
        y_fake = self.D(z_fake)
        loss_G = F.softmax_cross_entropy(y_fake, label_real)

        if chainer.config.train:
            update(self.G, self.opt_G, loss_G*6, is_unchain=True)

        return loss_D.data, loss_G.data, loss_REC.data
