#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, in_dim, n_hidden, output_dim, dropout_rate=0.2):
        """
        Args:
            in_dim: int, 输入维度, 28*28=784
            n_hidden: int, 隐层单元数
            output_dim: int, 输出维度
            dropout_rate: float
        """
        super(Generator, self).__init__()
        self.hidden_layer = nn.Linear(in_dim, n_hidden)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(n_hidden, output_dim)

    def forward(self, inputs):
        """
        Args:
            inputs: noise inputs, size=[bs, in_dim]

        Returns:
            outputs: generator outputs, size=[bs, out_dim]
        """
        logits = self.hidden_layer(inputs)
        logits = F.leaky_relu(logits, 0.1)
        logits = self.dropout_layer(logits)
        logits = self.output_layer(logits)
        outputs = F.tanh(logits)
        return outputs


class Discriminator(nn.Module):

    def __init__(self, in_dim, n_hidden):
        """
        Args:
            in_dim: int, 输入维度, 28*28=784
            n_hidden: int, 隐层单元数
        """
        super(Discriminator, self).__init__()
        self.hidden_layer = nn.Linear(in_dim, n_hidden)
        self.output_layer = nn.Linear(n_hidden, 1)

    def forward(self, inputs):
        """
        Args:
            inputs: size=[bs, in_dim]

        Returns:
            outputs: size=[bs]
        """
        logits = self.hidden_layer(inputs)
        logits = F.leaky_relu(logits, 0.1)
        logits = self.output_layer(logits)
        outputs = F.sigmoid(logits).view(-1)
        return outputs
