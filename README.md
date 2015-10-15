# gan_tutorial

Setup Torch
===
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash

git clone https://github.com/torch/distro.git ~/torch --recursive

cd ~/torch; ./install.sh

Train GAN on mnist
===
train_mnist.lua 

Train CGAN on mnist
===
train_mnist_conditional.lua 
