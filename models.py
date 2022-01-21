import torch.nn as nn
import torch.nn.functional as F

# split up the model from the pytorch classification tutorial on cifar10 https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# convolution portion
model1 = nn.Sequential(nn.Conv2d(3,6,5),
                       nn.ReLU(),
                       nn.MaxPool2d(2,2),
                       nn.Conv2d(6,16,5),
                       nn.MaxPool2d(2,2),
                       nn.ReLU()
                       )

# reduction portion
model2 = nn.Sequential(nn.Linear(16*5*5,120),
                       nn.ReLU(),
                       nn.Linear(120,84),
                       nn.ReLU())

# classification portion
model3 = nn.Sequential(nn.Linear(84,10))
