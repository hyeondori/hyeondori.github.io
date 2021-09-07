---
layout: single
title: "선형회귀 모델을 PyTorch로 가볍게 구현해보자"
search: true
excerpt: "PyTorch를 통한 모델링 기본 이해"
last_modified_at: 2021-09-07T16:57:00+09:00
toc: true
categories:
  - ML,DL
tags:
  - PyTorch
  - Linear Regression
---
*[본 포스팅은 김성훈 교수님의 PyTorch Zero To All Lecture을 보며 작성한 글입니다]*

**PyTorch**는 세가지의 리듬을 갖고 있다.

1. **Class와 변수**를 통해 **model을 디자인**한다.
2. PyTorch API를 통해 **loss와 optimizer**를 구성한다.
3. 순전파, 역전파, 업데이트의 **Training cycle**을 실행한다.

자 그럼 PyTorch로 모델을 구현해보자.

## Module Import


```python
from torch import nn
import torch
from torch import tensor
```

## 변수 선언


```python
x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])
```

## Step 1. Class 선언
Class 이름은 아무렇게나 지어도 된다. 여기서는 Model로 선언하겠다.
이번에는 Linear 모델을 활용할 것이기 때문에 `torch.nn.Linear()`로 선언한다.
괄호 안의 1, 1은 Input과 Output을 의미한다.
우리는 **하나의 x에서 하나의 y를 예측할 것이기 때문에 1, 1을 사용**한다.


```python
class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred
# our model
model = Model()
```

## Step 2. Loss와 Optimizer 구성
우리는 선형회귀 모델을 구현할 것이기 때문에 **MSE loss**를 활용할 것이다
**Optimizer**는 **Stochastic Gradient Descent**를 사용하여 쉽게 구현할 생각이다.


```python
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

## Step 3. Training
**1)순전파** 이후 **2)계산된 loss**를 통해 **3)역전파**로 이전 노드로 보내 가중치를 업데이트 하는 과정을 거쳐 각 노드의 가중치들을 업데이트할 것이다.
**epoch**는 이러한 과정을 몇번 반복할 것인지를 의미하며, 여기서는 500회 실행할 것이다.


```python
# Training loop
for epoch in range(500):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data)
    if epoch % 100 == 0:
        print(f'Epoch: {epoch} | Loss: {loss.item()} ')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

    Epoch: 0 | Loss: 47.002342224121094 
    Epoch: 100 | Loss: 0.08923976868391037 
    Epoch: 200 | Loss: 0.020983560010790825 
    Epoch: 300 | Loss: 0.004934028722345829 
    Epoch: 400 | Loss: 0.0011601777514442801 


훈련을 마친 후, Test를 위해 변수 hour_var을 선언한 후, 모델을 통해 예측을 진행해보자.


```python
# After training
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())
```

    Prediction (after training) 4 7.980875492095947


500회에 걸쳐 훈련을 진행했더니, 예측값이 8에 굉장히 근사했음을 볼 수 있다.
이와 같이 PyTorch를 통해 Linear Regression Model을 간단히 구현해볼 수 있었다.
처음부터 끝까지 한번 직접 코드를 따라해보면 PyTorch를 통한 모델 구현을 이해할 수 있을 것이다.
