---
layout: single
title: "다중 클래스 분류 문제를 위한 해법, Softmax Function"
search: true
excerpt: ""
last_modified_at: 2021-09-23T15:30:00+09:00
toc: true
categories:
  - ML,DL
tags:
  - PyTorch
  - Softmax
use_math: true

---

*[본 포스팅은 김성훈 교수님의 PyTorch Zero To All Lecture을 보며 작성한 글입니다]*

## 들어가기 전에...

앞선 포스팅에서는 Output size가 1인 경우에 대해서만 다뤘다. 그러나 우리 주변에는 Output size가 그 이상인 경우가 훨씬 많다. 아주 basic한 dataset으로 불려지는 MNIST 데이터셋 또한 0부터 9까지의 10가지 label을 갖고 있어 10개의 Output을 필요로 한다. 이 경우 우리는 Activation Function으로 Softmax함수를 활용하게 된다.

Softmax 함수는 다음과 같이 구성된다.

### Softmax 함수

$$
p_j=\frac{e^{z_j}}{\displaystyle\sum^K_{k=1}e^{z_j}}\\j = 1, 2, ..., K
$$

Softmax 함수의 작동 원리를 설명하는 것은 다소 길어질 수 있어 따로 정리하도록 하겠다.

중요한 점은 Softmax Function의 결과는 원하는 Output 결과 갯수만큼의 0과 1 사이의 값으로 떨어지게 되고, 이 값들의 합은 1이 된다.

Multi-class classification 문제를 해결하기 위해 softmax 함수와 함께 사용하는 Loss Function은 Cross Entropy Loss이다.



## Import Library


```python
# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
```


```python
# Training settings
batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training MNIST Model on {device}\n{"=" * 44}')
```

    Training MNIST Model on cuda
    ============================================

## Set Datasets

```python
# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=batch_size,
                               shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz


    2.3%
    
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ./mnist_data/MNIST\raw\train-images-idx3-ubyte.gz


    64.5%IOPub message rate exceeded.
    The Jupyter server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--ServerApp.iopub_msg_rate_limit`.
    
    Current values:
    ServerApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    ServerApp.rate_limit_window=3.0 (secs)
    
    100.0%


    Extracting ./mnist_data/MNIST\raw\train-images-idx3-ubyte.gz to ./mnist_data/MNIST\raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz


    102.8%
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ./mnist_data/MNIST\raw\train-labels-idx1-ubyte.gz
    Extracting ./mnist_data/MNIST\raw\train-labels-idx1-ubyte.gz to ./mnist_data/MNIST\raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ./mnist_data/MNIST\raw\t10k-images-idx3-ubyte.gz


​    
​    28.6%IOPub message rate exceeded.
​    The Jupyter server will temporarily stop sending output
​    to the client in order to avoid crashing it.
​    To change this limit, set the config variable
​    `--ServerApp.iopub_msg_rate_limit`.
​    
​    Current values:
​    ServerApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
​    ServerApp.rate_limit_window=3.0 (secs)

    ## Modeling


```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
```


```python
model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
```

## Training


```python
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)')
```


```python
if __name__ == '__main__':
    since = time.time()
    for epoch in range(1, 10):
        epoch_start = time.time()
        train(epoch)
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Training time: {m:.0f}m {s:.0f}s')
        test()
        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Testing time: {m:.0f}m {s:.0f}s')

    m, s = divmod(time.time() - since, 60)
    print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {device}!')
```

    Train Epoch: 1 | Batch Status: 0/60000 (0%) | Loss: 2.296072
    Train Epoch: 1 | Batch Status: 640/60000 (1%) | Loss: 2.295131
    Train Epoch: 1 | Batch Status: 1280/60000 (2%) | Loss: 2.301317
    Train Epoch: 1 | Batch Status: 1920/60000 (3%) | Loss: 2.314219
    Train Epoch: 1 | Batch Status: 2560/60000 (4%) | Loss: 2.298035
    ...
    Train Epoch: 9 | Batch Status: 55040/60000 (92%) | Loss: 0.043041
    Train Epoch: 9 | Batch Status: 55680/60000 (93%) | Loss: 0.035200
    Train Epoch: 9 | Batch Status: 56320/60000 (94%) | Loss: 0.110762
    Train Epoch: 9 | Batch Status: 56960/60000 (95%) | Loss: 0.101608
    Train Epoch: 9 | Batch Status: 57600/60000 (96%) | Loss: 0.086552
    Train Epoch: 9 | Batch Status: 58240/60000 (97%) | Loss: 0.021303
    Train Epoch: 9 | Batch Status: 58880/60000 (98%) | Loss: 0.202007
    Train Epoch: 9 | Batch Status: 59520/60000 (99%) | Loss: 0.113662
    Training time: 0m 11s
    ===========================
    Test set: Average loss: 0.0016, Accuracy: 9699/10000 (97%)
    Testing time: 0m 12s
    Total Time: 1m 45s
    Model was trained on cpu!


이번 포스팅은 필자가 softmax 함수에 대한 이해가 부족하여 여기서 마무리한다. 추후 이해가 더해진다면 해당 포스팅의 내용을 더욱 보충하도록 하겠다.

포스팅에 오류가 있거나, 덧붙일 사항이 있다면 언제든지 자유롭게 남겨주세요! 오늘도 행복하세요 :)
