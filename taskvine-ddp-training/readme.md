### Details about the experiment:
-----------------
The goal of this experiment is to prove that when we run two tasks on single node (note am not sure whethere this is a feature of taskvine or a bug) where it should basically throw error that already the GPU is occupied by one task, but what is happening is that the GPU is using timeslicing and performs the tasks.


#### Link I got the training model from
-----------------
https://github.com/NERSC/pytorch-examples/blob/main/models/resnet_cifar10.py
https://docs.pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet50-mnist.ipynb
