import torch.distributed.rpc
import torch
import torch.nn as nn
from models import model1,model2,model3
import torchvision
import torchvision.transforms as transforms
from torch.distributed.rpc import RRef
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer

class alice(object):
    def __init__(self,server,bob_model_rrefs):
        self.bob = server

        self.model1 = model1
        self.model2 = model3


        self.criterion = nn.CrossEntropyLoss()

        self.dist_optimizer=  DistributedOptimizer(
                    torch.optim.SGD,
                    list(map(lambda x: RRef(x),self.model2.parameters())) +  bob_model_rrefs +  list(map(lambda x: RRef(x),self.model1.parameters())),
                    lr=0.001,
                    momentum = 0.9
                )

        self.load_data()

    def train(self):
        running_loss = 0.0

        for i,data in enumerate(self.train_dataloader):
            inputs,labels = data

            with dist_autograd.context() as context_id:

                activation_alice1 = self.model1(inputs)
                activation_alice1 = torch.flatten(activation_alice1, 1)
                activation_bob = self.bob.rpc_sync().train(activation_alice1) #model(activation_alice1)
                activation_alice2 = self.model2(activation_bob)

                loss = self.criterion(activation_alice2,labels)

                # run the backward pass
                dist_autograd.backward(context_id, [loss])

                self.dist_optimizer.step(context_id)

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[x, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    def load_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 4

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        self.train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        self.test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)


class bob(object):
    def __init__(self):

        self.server = RRef(self)
        self.model = model2
        model_rrefs = list(map(lambda x: RRef(x),self.model.parameters()))

        self.alice = rpc.remote("alice", alice, (self.server,model_rrefs))

    def train_request(self):
        # call the train request from alice
        self.alice.rpc_sync(timeout=0).train()

    def train(self,x):
        return self.model(x)
