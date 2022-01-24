import logging

import torch.distributed.rpc
import torch
import torch.nn as nn
from models import *
import torchvision
import torchvision.transforms as transforms
from torch.distributed.rpc import RRef
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import logging
import os
from collections import Counter
from copy import deepcopy

class alice(object):
    def __init__(self,server,bob_model_rrefs,rank,args):
        self.client_id = rank
        self.epochs = args.epochs
        self.start_logger()

        self.bob = server

        self.model1 = model1()
        self.model2 = model3()

        self.criterion = nn.CrossEntropyLoss()

        self.dist_optimizer=  DistributedOptimizer(
                    torch.optim.SGD,
                    list(map(lambda x: RRef(x),self.model2.parameters())) +  bob_model_rrefs +  list(map(lambda x: RRef(x),self.model1.parameters())),
                    lr=args.lr,
                    momentum = 0.9
                )

        self.load_data(args)


    def train(self,last_alice_rref,last_alice_id):
        self.logger.info("Training")

        if last_alice_rref is None:
            self.logger.info(f"Alice{self.client_id} is first client to train")

        else:
            self.logger.info(f"Alice{self.client_id} receiving weights from Alice{last_alice_id}")
            model1_weights,model2_weights = last_alice_rref.rpc_sync().give_weights()
            self.model1.load_state_dict(model1_weights)
            self.model2.load_state_dict(model2_weights)


        for epoch in range(self.epochs):
            for i,data in enumerate(self.train_dataloader):
                inputs,labels = data

                with dist_autograd.context() as context_id:

                    activation_alice1 = self.model1(inputs)
                    activation_bob = self.bob.rpc_sync().train(activation_alice1) #model(activation_alice1)
                    activation_alice2 = self.model2(activation_bob)

                    loss = self.criterion(activation_alice2,labels)

                    # run the backward pass
                    dist_autograd.backward(context_id, [loss])

                    self.dist_optimizer.step(context_id)


    def give_weights(self):
        return [deepcopy(self.model1.state_dict()), deepcopy(self.model2.state_dict())]

    def eval(self):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.test_dataloader:
                images, labels = data
                # calculate outputs by running images through the network
                activation_alice1 = self.model1(images)
                activation_bob = self.bob.rpc_sync().train(activation_alice1)  # model(activation_alice1)
                outputs = self.model2(activation_bob)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.logger.info(f"Alice{self.client_id} Evaluating Data: {round(correct / total, 3)}")
        return correct, total

    def load_data(self,args):
        self.train_dataloader = torch.load(os.path.join(args.datapath ,f"data_worker{self.client_id}_train.pt"))
        self.test_dataloader = torch.load(os.path.join(args.datapath ,f"data_worker{self.client_id}_test.pt"))

        self.n_train = len(self.train_dataloader.dataset)
        self.logger.info("Local Data Statistics:")
        self.logger.info("Dataset Size: {:.2f}".format(self.n_train))
        self.logger.info(dict(Counter(self.test_dataloader.dataset[:][1].numpy().tolist())))

    def start_logger(self):
        self.logger = logging.getLogger(f"alice{self.client_id}")
        self.logger.setLevel(logging.INFO)

        format = logging.Formatter("%(asctime)s: %(message)s")

        fh = logging.FileHandler(filename=f"logs/alice{self.client_id}.log",mode='w')
        fh.setFormatter(format)
        fh.setLevel(logging.INFO)

        self.logger.addHandler(fh)

        self.logger.info("Alice is going insane!")



class bob(object):
    def __init__(self,args):

        self.server = RRef(self)
        self.model = model2()
        model_rrefs = list(map(lambda x: RRef(x),self.model.parameters()))

        self.alices = {rank+1: rpc.remote(f"alice{rank+1}", alice, (self.server,model_rrefs,rank+1,args)) for rank in range(args.client_num_in_total)}
        self.last_alice_id = None
        self.client_num_in_total  = args.client_num_in_total
        self.start_logger()

    def train_request(self,client_id):
        # call the train request from alice
        self.logger.info(f"Train Request for Alice{client_id}")
        if self.last_alice_id is None:
            self.alices[client_id].rpc_sync(timeout=0).train(None,None)
        else:
            self.alices[client_id].rpc_sync(timeout=0).train(self.alices[self.last_alice_id],self.last_alice_id)
        self.last_alice_id = client_id

    def eval_request(self):
        self.logger.info("Initializing Evaluation of all Alices")
        total = []
        num_corr = []
        check_eval = [self.alices[client_id].rpc_async(timeout=0).eval() for client_id in
                      range(1, self.client_num_in_total + 1)]
        for check in check_eval:
            corr, tot = check.wait()
            total.append(tot)
            num_corr.append(corr)

        self.logger.info("Accuracy over all data: {:.3f}".format(sum(num_corr) / sum(total)))

    def train(self,x):
        return self.model(x)

    def start_logger(self):
        self.logger = logging.getLogger("bob")
        self.logger.setLevel(logging.INFO)

        format = logging.Formatter("%(asctime)s: %(message)s")

        fh = logging.FileHandler(filename="logs/bob.log", mode='w')
        fh.setFormatter(format)
        fh.setLevel(logging.INFO)

        self.logger.addHandler(fh)
        self.logger.info("Bob Started Getting Tipsy")
