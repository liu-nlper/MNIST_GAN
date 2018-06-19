#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from utils import init_dataloader, show_image_batch
from model import Generator, Discriminator


# args
n_hidden = 128
noise_dim = 100
img_dim = 28 * 28
batch_size = 256
n_epoch = 100
lr = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# init datasets
data_loader = init_dataloader(batch_size=batch_size)

# init generator
generator = Generator(noise_dim, n_hidden, img_dim).to(device)
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
print(generator)

# init discriminator
discriminator = Discriminator(img_dim, n_hidden).to(device)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)
print(discriminator)

# define loss function
loss_function = nn.BCELoss()

# 测试使用...
noise4test = torch.randn((1, noise_dim)).to(device)
g_test_outputs = []

# train...
for epoch in range(n_epoch):
    loss_discriminator_all, loss_generator_all = [], []
    print('Epoch {0} / {1}:'.format(epoch+1, n_epoch))
    images4show = None
    for batch in data_loader:

        # real data
        inputs_real, labels_real = batch[0], batch[1]
        inputs_real = inputs_real.view(inputs_real.size(0), -1)  # [bs, 28*28]
        inputs_real = inputs_real.to(device)

        # feak data
        inputs_noise = torch.randn((batch_size, noise_dim)).to(device)

        # 训练判别器
        discriminator.zero_grad()
        g_outputs = generator(inputs_noise)
        if images4show is None:
            images4show = g_outputs.data
        d_outputs_real = discriminator(inputs_real)
        d_outputs_fake = discriminator(g_outputs)
        loss_real = loss_function(
            d_outputs_real, torch.ones(d_outputs_real.size(0)).to(device))
        loss_fake = loss_function(
            d_outputs_fake, torch.zeros(d_outputs_fake.size(0)).to(device))
        loss_discriminator = loss_real + loss_fake
        loss_discriminator_all.append(loss_discriminator.item())
        loss_discriminator.backward(retain_graph=True)
        optimizer_d.step()

        # 训练生成器
        generator.zero_grad()
        loss_generator = loss_function(
            d_outputs_fake, torch.ones(d_outputs_fake.size(0)).to(device))
        loss_generator_all.append(loss_generator.item())
        loss_generator.backward()
        optimizer_g.step()

    # 测试用例
    g_test_output = generator(noise4test)
    g_test_outputs.append(g_test_output.data)

    print('discriminator loss: {0:.2f}, generator loss: {1:.2f}'.format(
        np.mean(loss_discriminator_all), np.mean(loss_generator_all)))

# 测试生成器在不同迭代次数下，对用一个输入(noise)产生的输出
g_test_outputs = torch.cat(g_test_outputs, 0)
show_image_batch(g_test_outputs, show=True)
