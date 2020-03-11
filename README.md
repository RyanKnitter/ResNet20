# ResNet20

In the [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) paper, the authors describe networks that were tested on the CIFAR10 dataset. Models were built with 20, 32, 44, and 56 layers. In this repository, the 20-layer model was recreated in [pytorch](pytorch.org). The purpose of this repository is to explicitly show each layer in the network so those who are new to this type of model can get a better understanding of how they work.

The ResNet20Parms.pt file contains the parameters for a 20-layer model that was trained on the CIFAR10 dataset. The model with these parameters accurately classifies the images about 80% of the time.

Here is some sample code for how to initialize a model with the pre-trained parameters.
<pre><code>model = ResNetModel.ResNet()
model.load_state_dict(torch.load("ResNet20Parms.pt"))
model.eval()
</code></pre>
