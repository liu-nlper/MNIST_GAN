Mnist GAN PyTorch版本，详细介绍可以参考知乎专栏[机器不学习](https://zhuanlan.zhihu.com/zhaoyeyu)的文章[生成对抗网络（GAN）之MNIST数据生成](https://zhuanlan.zhihu.com/p/28057434)。

## 1. run

clone至本地：

    git clone https://github.com/liu-nlper/MNIST_GAN

安装依赖项：

    $ cd MNIST_GAN/code
    $ sudo pip3 install -r ./code/requirements.txt --upgrade  # for all user
    $ pip3 install -r ./code/requirements.txt --upgrade --user  # for current user

运行：

    $ python3 main.py  # cpu
    $ CUDA_VISIBLE_DEVICES=0 python3 main.py  # gpu

运行结果：

**图1.**： 不同迭代次数下生成器对同一个噪声输入产生的输出

![mnist.png](https://github.com/liu-nlper/MNIST_GAN/blob/master/mnist_gan/mnist.png)

**图2.**： 迭代100次后生成器的输出

![mnist.png](https://github.com/liu-nlper/MNIST_GAN/blob/master/mnist_gan/mnist2.png)

从图中可以看出，由于生成器是一个两层的MLP，学习能力较弱，所以生成的手写数字图像质量不高。

注：noise输入是随机产生，所以每次运行结果都会不一样。

## 2. requirements

 - python3
   - matplotlib
   - numpy
   - torch==0.4.0p
   - torchvision