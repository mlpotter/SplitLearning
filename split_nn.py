from data_entities import alice,bob
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import os
import argparse

def init_env():
    print("Initialize Meetup Spot")
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ["MASTER_PORT"] = "5689"

def example(rank,world_size,args):
    init_env()
    if rank == 0:
        rpc.init_rpc("bob", rank=rank, world_size=world_size)

        BOB = bob()

        for iter in range(args.iterations):
            BOB.train_request()
            BOB.eval_request()

        rpc.shutdown()
    else:
        rpc.init_rpc("alice", rank=rank, world_size=world_size)
        rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split Learning Initialization')
    parser.add_argument('--iterations',type=int,default=5,help='The number of iterations to communication between clients and server')

    args = parser.parse_args()

    world_size = 2
    mp.spawn(example,
             args=(world_size,args),
             nprocs=world_size,
             join=True)