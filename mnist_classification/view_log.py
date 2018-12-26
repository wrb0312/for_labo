"""
Edit by Keisuke Oyamada @2018/12/25.
For training bachelor.
"""

import json
import matplotlib.pyplot as plt
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="MNIST example for test")
    parser.add_argument("--json_path", "-j", help="model path", required=True, type=str)
    parser.add_argument("--save_path", "-s", help="data save path", required=True, type=str)

    args = parser.parse_args()

    json_path = args.json_path
    save_path = args.save_path

    print("MNIST example for log")
    print("json path: {}".format(json_path))
    print("save path: {}".format(save_path))

    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []
    epoch = []
    with open(json_path) as f:
        for i in f:
            j = json.loads(i)
            loss_train.append(j["loss_train"])
            loss_test.append(j["loss_test"])
            acc_train.append(j["acc_train"]*100)
            acc_test.append(j["acc_test"]*100)
            epoch.append(j["epoch"])

    plt.plot(epoch, loss_train, label="loss_train", marker="*")
    plt.plot(epoch, loss_test, label="loss_test", marker="*")
    plt.xticks(epoch[::3])
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(os.path.join(save_path, "loss.jpg"))
    plt.close()

    plt.plot(epoch, acc_train, label="acc_train", marker="*")
    plt.plot(epoch, acc_test, label="acc_test", marker="*")
    plt.xticks(epoch[::3])
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("acc [%]")
    plt.savefig(os.path.join(save_path, "acc.jpg"))
    plt.close()


if __name__ == "__main__":
    main()
