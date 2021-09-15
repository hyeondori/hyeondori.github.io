---
layout: single
title: "PyTorch를 통해 모델을 더 넓고 깊게 만들어보자"
search: true
excerpt: "딥러닝 입문"
last_modified_at: 2021-09-15T10:50:00+09:00
toc: true
categories:
  - ML,DL
tags:
  - PyTorch
---

*[본 포스팅은 김성훈 교수님의 PyTorch Zero To All Lecture을 보며 작성한 글입니다]*

## 들어가기 전에...
지금까지 우리는 상당히 간단한 1x1의 모델을 만들어왔다.

당연하겠지만 이런 형태의 모델은 제대로 된 값을 얻는 것에 무리가 있다.

이번 포스팅에서는 이 모델을 좀 더 넓고 깊게 만드는 방법에 대해 PyTorch 코드로 함께 접근해보겠다.

지금까지 우리는 아래와 같이 열이 하나인 단순한 x값을 Input data로 넣어왔다.

```python
x_data = [[1.0], [2.0], [3.0], [4.0]]
y_data = [[0.], [0.], [1.], [1.]]
```

그렇다면, 아래와 같은 형태의 x값을 계산하려면 어떻게 해야할까?

```python
x_data = [[2.1, 0.1],
          [4.2, 0.8],
          [3.1, 0.9],
          [3.3, 0.2]]
y_data = [[0.0],
          [1.0],
          [0.0],
          [1.0]]
```

우리는 여기서 행렬곱을 활용한다.

x_data가 4x2 형태이고, 얻어야 할 y_data가 4x1 형태이면 이를 얻기 위해 w는 2x1 형태의 행렬이면 된다!

아주 간단하므로 따로 설명없이 넘어가겠다.

그렇다면 여기서 4의 위치에는 n개의 갯수만 오면 될 것이고, 그것이 몇개 존재하던지 상관없이 w 행렬을 통해 n x ? 형태의 원하는 형태의 값을 얻어낼 수 있음을 알 수 있다.

이러한 방식으로 우리는 모델을 넓게 만들 수 있다.

그렇다면 모델을 깊게 만드는 것은 어떻게 할 수 있을까?

딥러닝을 조금 겉핥기로 들어본 사람은 '레이어'라는 개념을 들어본 적이 있을 것이다.

레이어를 여러개 쌓아 층을 깊게 쌓으면서 우리는 이를 보고 'Deep'하다고 표현을 하고, 여기서 'Deep Learning'이란 느낌있는 말이 탄생한 것이다.

코드로 보면 무슨 말인지 느낌이 올 것이다.

```python
sigmoid = torch.nn.sigmoid()

11 = torch.nn.Linear(2, 4)
12 = torch.nn.Linear(4, 3)
13 = torch.nn.Linear(3, 1)

out1 = sigmoid(11(x_data))
out2 = sigmoid(12(out1))
y_pred = sigmoid(13(out2))
```

여기서 유의할 점은 Input size와 Output size이다.
이전 레이어의 output이 4라면 다음 레이어의 input은 4여야 한다는 의미이다.

그러나 지금까지 활용한 Activation Function: sigmoid 함수의 경우, 이렇게 레이어를 여러겹 쌓았을 때 Vanishing Gradient Problem이 발생한다.
이 문제에 대해서는 다음에 좀 더 자세히 따로 다뤄보도록 하겠다.

앞으로는 이를 해결하기 위해 ReLU함수로 이를 대체할 것이다. ReLU함수는 딥러닝이 다시 부활할 수 있도록 만든 일등공신이라고 할 수 있다.

이러한 Activation Functions은 이외에도 정말 많으니 앞으로 다양하게 어떤 장점과 단점이 있는지 알아보면서 공부하도록 하자.


## Import Library


```python
from torch import nn, optim, from_numpy
import numpy as np
```

이번 포스팅부터는 실 데이터를 활용하여 모델링을 할 것이다.


```python
xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
```

x값과 y값이 모두 담긴 csv파일 하나를 읽어와 numpy.float32 형태로 xy 변수에 할당한다.


```python
x_data = from_numpy(xy[:, 0:-1])
y_data = from_numpy(xy[:, [-1]])
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')
```

    X's shape: torch.Size([759, 8]) | Y's shape: torch.Size([759, 1])


이를 x값과 y값으로 나누어 torch형태로 할당한다.
우리는 X값의 shape을 통해 8개의 feature를 가진 759개의 데이터가 있음을 알 수 있다.

이를 통해 모델을 설계하면 다음과 같이 레이어를 쌓을 수 있다.


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

## 모델 정의


```python
# our model
model = Model()
```

## Loss함수와 optimizer 정의(PyTorch API)


```python
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

## Training


```python
# Training loop
for epoch in range(100):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch + 1}/100 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

    Epoch: 1/100 | Loss: 0.9357
    Epoch: 11/100 | Loss: 0.7510
    Epoch: 21/100 | Loss: 0.6834
    Epoch: 31/100 | Loss: 0.6591
    Epoch: 41/100 | Loss: 0.6502
    Epoch: 51/100 | Loss: 0.6469
    Epoch: 61/100 | Loss: 0.6457
    Epoch: 71/100 | Loss: 0.6452
    Epoch: 81/100 | Loss: 0.6450
    Epoch: 91/100 | Loss: 0.6449

