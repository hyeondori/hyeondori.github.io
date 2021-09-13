---
layout: single
title: "로지스틱 회귀 모델을 PyTorch로 가볍게 구현해보자"
search: true
excerpt: "PyTorch를 통한 모델링 기본 이해"
last_modified_at: 2021-09-13T17:07:00+09:00
toc: true
categories:
  - ML,DL
tags:
  - PyTorch
  - Logistic Regression

---

*[본 포스팅은 김성훈 교수님의 PyTorch Zero To All Lecture을 보며 작성한 글입니다]*

우리의 인생은 항상 선택의 연속이며 우리는 선택의 결과에 늘 큰 영향을 받는다.
전문용어로는 의사결정이라 이르며, 경영자의 의사결정에 의해 한 회사가 엄청난 성공을 거두기도 존폐의 기로에 서기도 한다.
그렇기에 0과 1로 결정하는 `binary solution`을 만드는 것은 굉장히 중요하다.
어떠한 의사결정에 중요한 단서를 제시하기 때문이다.

다시 좀 더 쉽게 접근하여 우리가 경험한 인생으로 돌아와보자.

* 시험 공부를 할 때, 몇시간을 들이면 합격할 것인가? 떨어질 것인가?
* 축구 경기에서 이길 것인가? 질 것인가?
* 좋아하는 사람이 생겼을 때, 고백할 것인가? 말 것인가?
* 누군가 이러한 결과를 미리 알려준다면 인생에 큰 고민들을 꽤 날려버릴 수 있을 것이다.

우리는 컴퓨터라는 기가 막힌 도구가 우리의 의사결정을 도와주기 위해 하나의 함수를 활용할 것이다.
~~지금 이 글을 읽고 있을 사람은 딥린이일 것이라 믿어 의심치 않고, 함수가 등장했다고 절대 겁먹지 않았으면 한다.~~
지금 설명할 `Sigmoid` 함수, 앞으로 나올 `ReLU` 함수 등은 컴퓨터가 계산한 무언가를 다시 한번 이해하기 좋게 변환하는 과정을 위해 필요하다 라고 생각하고 넘어가면 좋겠다.

오늘 설명할 함수는 `Sigmoid` 함수이다.

한국어로는 시그모이드 함수라고 쓰며, `Logistic` 함수라고도 한다. 해당 함수는 다음과 같이 구성된다.
$$
σ(x)=\frac{1}{1+e^{-x}}
$$
~~자연로그 e를 보고 어지럽다면 당신은 문과 출신이다. 하지만 괜찮다. 필자도 똑같다.~~

<img src="\assets\images\typora-user-images/sigmoid.png" alt="sig" style="zoom:67%;" />

`Sigmoid` 함수 양 끝을 보면 0과 1로 수렴하는 것을 볼 수 있다. 왜 갑자기 이 함수가 등장했는지 그림으로 봤으니 와닿을 것이라 생각한다.
전통적으로 연구자들은 컴퓨터가 계산한 무언가를 0과 1의 결과값을 얻기 위해 이와 같은 함수를 활용해왔다.
딥러닝의 역사에 대해 조금 공부해본다면 딥러닝의 빙하기가 왜 찾아오게 되었는지 또한 이 함수와 연관이 있다. 이어서 `ReLU`함수, `TanH`함수 등과 관련된 포스팅을 하게 되면 이에 관해 언급하겠다.

다시 첫 번째 질문으로 돌아와보자, 시험공부를 할 때 몇 시간을 하면 합격할 것인가?

당신이 위 `Sigmoid`함수에서 0의 결과를 뱉는 x값만큼 공부를 했다면 결과는 0, 시험에 떨어질 것이고 1의 결과를 뱉는 x값만큼 공부를 했다면 시험에 합격할 것이다.

물론 우리의 인생은 하나의 x값, 여기서는 시험 합격을 위해 공부시간 하나만으로 해결되지 않는다.

하지만 이러한 간단한 생각을 통해 **Logistic Regression**이 탄생했다.

우리는 추후에 이 생각을 베이스로 x 변수값을 추가해보기도 하고 오히려 줄여보기도 하며 더 좋은 모델을 만들 수 있을 것이다.

그럼 바로 `PyTorch`에서 `Sigmoid`함수가 어떻게 쓰이는지 코드와 함께 보자.


```python
import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
```

위의 코드를 중 아래의 두번째 줄에서 `PyTorch`에서 친절히 `sigmoid` 함수를 사용할 수 있도록 만들어 주었고, 이를 활용할 수 있음을 확인할 수 있다.

그리고 이 결과가 이상할 때, 우리는 Loss를 크게 주는 Loss Function을 뒤에 붙여 학습을 반복한다고 앞선 포스팅에서 배웠다. 혹시 모르겠다면 앞의 포스팅을 보고 오자!
앞선 포스팅에서는 `MSE Loss`를 활용했지만, 오늘은 `BCE Loss`(Binary Cross Entropy Loss)를 활용할 것이다.

`PyTorch`로 구현하면 다음과 같다.


```python
criterion = torch.nn.BCELoss(reduction='mean')
```

## Import Libarary


```python
from torch import tensor
from torch import nn
from torch import sigmoid
import torch.nn.functional as F
import torch.optim as optim
```

## Datasets

앞선 포스팅과 달라진 것은 y값에 0 또는 1이 들어갔다는 점이다.

왜냐하면 이건 **Logistic Regression**이기 때문이다.


```python
# Training data and ground truth
x_data = tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = tensor([[0.], [0.], [1.], [1.]])
```


```python
class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data.
        """
        y_pred = sigmoid(self.linear(x))
        return y_pred
```


```python
# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

1000번의 `Epoch`에 걸쳐 훈련이 진행되고, 훈련이 진행될수록 Loss율이 떨어지는 것을 확인할 수 있다.


```python
# Training loop
for epoch in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    if epoch % 100 == 0:
        print(f'Epoch {epoch + 1}/1000 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

    Epoch 1/1000 | Loss: 0.6502
    Epoch 101/1000 | Loss: 0.6231
    Epoch 201/1000 | Loss: 0.5986
    Epoch 301/1000 | Loss: 0.5758
    Epoch 401/1000 | Loss: 0.5546
    Epoch 501/1000 | Loss: 0.5348
    Epoch 601/1000 | Loss: 0.5163
    Epoch 701/1000 | Loss: 0.4990
    Epoch 801/1000 | Loss: 0.4829
    Epoch 901/1000 | Loss: 0.4678


훈련된 모델에 따르면, 1시간을 공부하면 시험에 떨어질 것이고 7시간을 공부하면 시험에 합격할 것이라고 한다.

우리 모두 열심히 공부합시다!


```python
# After training
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')
hour_var = model(tensor([[1.0]]))
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(tensor([[7.0]]))
print(f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}')
```


    Let's predict the hours need to score above 50%
    ==================================================
    Prediction after 1 hour of training: 0.3779 | Above 50%: False
    Prediction after 7 hours of training: 0.9714 | Above 50%: True



포스팅에 관해 추가적인 질문이나 오류가 있다면 아래 댓글로 의견 남겨 주시기 바랍니다 :)
