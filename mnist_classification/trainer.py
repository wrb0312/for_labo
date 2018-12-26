import chainer
import numpy as np
from tqdm import tqdm
import math
import os
from chainer import serializers
import chainer.functions as F
from reporter import reporter
from chainer import dataset


def update(model, opt, loss):
    model.cleargrads()
    loss.backward()
    opt.update()
    loss.unchain_backward()


class Trainer:
    def __init__(self, **kwargs):
        self.model = kwargs.pop("model")
        self.opt = kwargs.pop("opt")
        self.epochs = kwargs.pop("epochs")
        self.save_path = kwargs.pop("save_path")
        self.iter_train, self.iter_test = kwargs.pop("iterator")
        N, N_test = kwargs.pop("data_size")
        batch_size, batch_size_test = kwargs.pop("batch_size")
        self.xp = self.model.xp
        self.keys_general = ["epoch"]
        self.keys_train = ["loss_train",
                           "acc_train"]
        self.keys_test = ["loss_test",
                          "acc_test"]
        self.report = reporter(self.keys_general, self.keys_train, self.keys_test, N, N_test, self.save_path)
        self.N, self.N_test = math.ceil(N / batch_size), math.ceil(N_test / batch_size_test)

    def __call__(self):
        for epoch in range(1, self.epochs+1):
            self.report.epoch = epoch

            ### train ###
            num_train = 0
            bar = tqdm(desc="Training", total=self.N, leave=False)
            for d in self.iter_train:
                x, y = dataset.concat_examples(d)
                loss, acc = self.forward(x, y)
                self.report.loss_train += float(loss * len(x))
                self.report.acc_train += float(acc * len(x))
                num_train += len(x)

                bar.set_description("epoch: {}, loss: {:.6f}, acc: {:.6f}".format(
                    epoch,
                    self.report.loss_train / num_train,
                    self.report.acc_train / num_train),
                    refresh=False)
                bar.update()
            bar.close()

            ### test ###
            num_test = 0
            bar_test = tqdm(desc="Test", total=self.N_test, leave=False)
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                for d in self.iter_test:
                    x, y = dataset.concat_examples(d)
                    loss, acc = self.forward(x, y)
                    self.report.loss_test += float(loss * len(x))
                    self.report.acc_test += float(acc * len(x))
                    num_test += len(x)
                    bar_test.update()
            bar_test.close()

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            self.report(self.keys_general, self.keys_train, self.keys_test)

            self.iter_train.reset()
            self.iter_test.reset()
            self.report.init_log()

            serializers.save_npz(os.path.join(
                self.save_path, "mnist_ex.model"), self.model)
            serializers.save_npz(os.path.join(
                self.save_path, "mnist_ex.optimizer"), self.opt)

    def forward(self, x, y):
        x_v = chainer.Variable(self.xp.asarray(x))
        y_v = chainer.Variable(self.xp.asarray(y))
        y_h = self.model(x_v)
        loss = F.softmax_cross_entropy(y_h, y_v)
        acc = F.accuracy(y_h, y_v)
        if chainer.config.train:
            update(self.model, self.opt, loss)

        return loss.data, acc.data
