#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import itertools
import numpy as np

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms


def init_dataloader(batch_size=64):

    # Training dataset
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root='./data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ])), batch_size=batch_size, shuffle=True, num_workers=4
    )

    return data_loader


def show_image(image):
    """
    绘制图像
    """
    plt.imshow(image.reshape(28, 28), cmap='Greys_r')
    plt.show()


def show_image_batch(images, show=False, path='./mnist.png'):
    """
    Args:
        images: size=[nb, 28*28]
        show: bool, 是否画出图像，若是False，则保存
        path: str, 保存路径
    """
    width = math.ceil(np.sqrt(images.size(0)))
    fig, ax = plt.subplots(width, width, figsize=(width, width))
    for i, j in itertools.product(range(width), range(width)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(width*width):
        if k >= images.size(0):
            break
        i, j = k // width, k % width
        ax[i, j].cla()
        ax[i, j].imshow(images[k].view(28, 28), cmap='Greys_r')
    plt.savefig(path)
    if show:
        plt.show()


def demo():
    import matplotlib.pyplot as plt

    data_loader = init_dataloader()
    for data in data_loader:
        # inputs.size=[bs, 1, 28, 28], labels.size=[bs]
        inputs, labels = data[0], data[1]
        print(inputs.size())
        print(labels.size())

        plt.imshow(inputs[0].reshape(28, 28), cmap='Greys_r')
        plt.show()
        exit()


if __name__ == '__main__':
    demo()
