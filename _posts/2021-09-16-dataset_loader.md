---
layout: single
title: "PyTorch의 핵심, DataLoader"
search: true
excerpt: "데이터를 더욱 효율적으로 다루기 위한 방법"
last_modified_at: 2021-09-16T13:36:00+09:00
toc: true
categories:
  - ML,DL
tags:
  - PyTorch

---

*[본 포스팅은 김성훈 교수님의 PyTorch Zero To All Lecture을 보며 작성한 글입니다]*

## 들어가기 전에...

앞선 포스팅에서 다룬 데이터셋은 데이터셋 크기가 겨우 700개가 조금 넘었기에 통째로 잡아 넣어도 큰 문제가 발생하진 않았다.

그러나 만약 이 크기가 1억건 정도 넘어가는 빅데이터 크기가 된다면, 효율성 부분에서 분명히 문제가 발생할 것이다. 우리는 이를 해결하기 위해 `Batch`를 활용한다.

Batch size에 따라 데이터셋을 잘게 쪼개서 접근하는 방법이다. 

우리는 모델에 `Batch`를 반복적으로 넣고 학습하여 모델을 업데이트시키는 방향으로 모델링을 구성할 것이다.

Batch를 설정할 때 구성하는 요소는 3가지인데, 다음과 같다.

* `one epoch`: 순전파와 역전파가 전체(모든 pass의 합) 트레이닝 셋에 대해 한회차 일어나는 과정
* `batch size`: 순전파와 역전파 한번의 pass로 정의할 크기. batch size가 클수록 메모리를 많이 잡아먹는다.
* `number of iterations`: pass들의 갯수

다시 말로 풀어 설명하면, 하나의 pass는 순전파 한번 역전파 한번을 의미하고 batch size는 이 크기를 결정짓는 것을 의미한다.

이러한 구조를 이해하는 것은 중요하지만, 구현하는 것은 크게 어렵지 않다.

`PyTorch`에서는 `DataLoader`라는 것을 통해 이를 상당히 쉽고 간편하게 제공한다.

`DataLoader`는 `iteration`을 정의하여 `batch size`를 결정하는 방식으로 위의 방법을 쉽게 해결한다.



## Import Library


```python
# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
import numpy as np
```

## Custom DataLoader


```python
class DiabetesDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv.gz',
                        delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, [-1]])

    # return one item on the index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # return the data length
    def __len__(self):
        return self.len
```

dataset을 dataloader로 올리는데 적합하도록 class를 정의한 후


```python
dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          pin_memory=True)
```

DataLoader를 통해 batch size와 shuffle 등을 정의했다.

~~여기서 필자는 김성훈 교수님의 github에 올라온대로 코드를 작성했더니 Runtime Error가 발생해서 num_workers 대신에 pin_memory=True를 활용하였다. 이는 cuda에서 훈련을 진행함을 의미한다.~~

이후 필자가 PyTorch 세팅을 컴퓨터 사양에 맞게 제대로 하지 않은 것임을 깨달았다. 언젠가 PyTorch를 제대로 세팅하는 방법까지 알아보자.


```python
class Model(nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred
```

이전과 동일하게 모델을 정의하였다.


```python
# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(inputs)

        # Compute and print loss
        loss = criterion(y_pred, labels)
        print(f'Epoch {epoch + 1} | Batch: {i+1} | Loss: {loss.item():.4f}')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

    Epoch 1 | Batch: 1 | Loss: 22.3346
    Epoch 1 | Batch: 2 | Loss: 22.3792
    Epoch 1 | Batch: 3 | Loss: 15.3712
    ...
    Epoch 2 | Batch: 22 | Loss: 22.3706
    Epoch 2 | Batch: 23 | Loss: 22.1596
    Epoch 2 | Batch: 24 | Loss: 15.9283

앞서 정의한 train_loader로 학습을 진행하고 test_loader로 성능을 평가하는 과정을 반복하며, Loss가 점점 떨어지는 것을 볼 수 있다.

앞선 포스팅에 비해 훨씬 큰 데이터셋을 다룸에도 불구하고 DataLoader를 통해 효율적으로 모델에 학습이 되었다.

어서 빨리 딥러닝 모델을 자유자재로 다룰 수 있는 사람이 되기를 소망한다 :)
