# Split-Learning on Heterogenous Distributed MNIST

## Bob (coordinator)
Bob consists of two main functions 
1. *Train Request* 
   1. Request for Alice_x to update model weights to last trained Alice_x' weights.
   2. Perform flow of forward and backward pass in figure below for N*Batches iterations.
   3. Round robin fashion request for next Alice_x'' to begin training.
2. *Evaluation Request*
   1. Request for each Alice_x to perform evaluation on the test set.
   2. Aggregate results of each Alice_x and log the overall performance.
   
Details from https://dspace.mit.edu/bitstream/handle/1721.1/121966/1810.06060.pdf?sequence=2&isAllowed=y .

![Alt text](imgs/split_nn.PNG?raw=true  "Decentralized Split Learning Architecure")


Example run:
```python split_nn.py --epochs=2 --iterations=2 --world_size=5```

```
Split Learning Initialization

optional arguments:
  -h, --help            show this help message and exit
  --world_size WORLD_SIZE
                        The world size which is equal to 1 server + (world size - 1) clients
  --epochs EPOCHS       The number of epochs to run on the client training each iteration
  --iterations ITERATIONS
                        The number of iterations to communication between clients and server
  --batch_size BATCH_SIZE
                        The batch size during the epoch training
  --partition_alpha PARTITION_ALPHA
                        Number to describe the uniformity during sampling (heterogenous data generation for LDA)
  --datapath DATAPATH   folder path to all the local datasets
  --lr LR               Learning rate of local client (SGD)
```