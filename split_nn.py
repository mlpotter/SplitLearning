from data_entities import alice,bob
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import os
import argparse

def init_env():
    print("Initialize Meetup Spot")
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ["MASTER_PORT"] = "5689"

def example(rank,world_size):
    init_env()
    if rank == 0:
        rpc.init_rpc("bob", rank=rank, world_size=world_size)

        BOB = bob()

        for iter in range(3):
            BOB.train_request()

        rpc.shutdown()
    else:
        rpc.init_rpc("alice", rank=rank, world_size=world_size)
        rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split Learning Initialization')

    world_size = 2
    mp.spawn(example,
             args=(world_size,),
             nprocs=world_size,
             join=True)