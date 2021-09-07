---
layout: single
title: "역전파와 Autograd를 PyTorch로 가볍게 구현해보자"
search: true
excerpt: "역전파와 연쇄 법칙 이해"
last_modified_at: 2021-09-06T13:33:00+09:00
toc: true
categories:
  - ML/DL
tags:
  - PyTorch
  - Back Propagation
  - Autograd
---
# 역전파Back-propagation and Autograd



*[본 포스팅은 김성훈 교수님의 PyTorch Zero To All Lecture을 보며 작성한 글입니다]*



신경망 이론을 활용한 딥러닝을 접하게 되면 다음과 같은 그림을 자주 접하게 된다. 해당 그림은 그냥 보기에도 복잡하지만, 실제로 그림과 똑같이 컴퓨터로 구현하려면 여간 까다로운 것이 아니다.



<img src="\assets\images\typora-user-images\image-20210907141649480.png" alt="image-20210907141649480" style="zoom: 67%;" />



이를 조금 더 쉽게 구현하기 위해 Computational graph와 미적분학의 Chain Rule을 활용한다.



## Chain Rule&Backward propagation



Chain Rule에 대해 설명하기 위해 Chain Rule의 정의부터 살펴보자.



Chain Rule은 한국어로 연쇄 법칙이라 일컬으며, 함수의 합성에 대한 미분(또는 도함수)에 대한 공식이다. 이를 공식으로 표현하면 다음과 같이 표현할 수 있다.



![{\displaystyle (f\circ g)'(x_{0})=f'(g(x_{0}))g'(x_{0})}](https://wikimedia.org/api/rest_v1/media/math/render/svg/9a4a9d811065100c7802cdd56ee5b72caf17e72c)



당장 미분만 보면 골머리가 아파올 당신을 위해 빠르게 설명하자면(필자도 오랜만에 미분을 봐서 어질어질하다. 얼른 다시 익숙해지길), 함수 {\displaystyle g}![g](https://wikimedia.org/api/rest_v1/media/math/render/svg/d3556280e66fe2c0d0140df20935a6f057381d77)가 {\displaystyle x_{0}}![x_0](https://wikimedia.org/api/rest_v1/media/math/render/svg/86f21d0e31751534cd6584264ecf864a6aa792cf)에서 미분 가능하며 함수 {\displaystyle f}![f](https://wikimedia.org/api/rest_v1/media/math/render/svg/132e57acb643253e7810ee9702d9581f159a1c61)가 {\displaystyle g(x_{0})}![{\displaystyle g(x_{0})}](https://wikimedia.org/api/rest_v1/media/math/render/svg/2a4919ce8109c528145f19e09f5a07f5e9671956)에서 미분 가능할 때 {\displaystyle f\circ g}![f\circ g](https://wikimedia.org/api/rest_v1/media/math/render/svg/b2f61ca7838709fbae07dce9c0d513770f10cfae)는 {\displaystyle x_{0}}![x_0](https://wikimedia.org/api/rest_v1/media/math/render/svg/86f21d0e31751534cd6584264ecf864a6aa792cf)에서 미분 가능하다는 의미이다.



자, 어질어질한 당신을 위해 한번 더 설명하면 합성 함수 내의 각 구성요소가 각각 바로 아래 연계된 함수 혹은 변수에 대해 미분이 가능하면, 합성 함수는 Chain Rule에 의해 미분값을 구할 수 있다는 의미이다. 이를 라이프니츠 표기법으로 표현하면 다음과 같다.



![{\displaystyle {\frac {dy}{dx}}={\frac {dy}{du}}\cdot {\frac {du}{dx}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/955d845cb30bede7b50f3b9bef5e07e613e4373f)



이러한 Chain Rule을 쓰는 이유는 결국 Gradient of Loss를 구하기 위함인데, 이러한 계산을 하나하나 해보면서 실습하는 것은 물론 알고리즘을 이해하는데 도움이 되지만, **걱정하지 마라. 사실 이러한 계산을 모두 할 줄 알 필요는 없다. 우리한텐 컴퓨터가 있다.** 



~~농담반 진담반이지만 딥러닝에 진심이라면 한번쯤 계산해보는 것이 좋다. 이와 관련된 예제는 찾아보면 금방 나오니 한번 해보길 권장한다. 그러나, 이번 포스팅은 역전파와 Chain Rule을 PyTorch를 통해 구현까지 해보는 것이 목적이기 때문에, 개념만 짚고 넘어가겠다.~~



결론적으로 얘기하면 Backward propagation을 통해 어떤 값을 받기 전에 Local gradient를 미리 계산해 놓을 수 있고 이를 조합하여 Chain Rule에 의해 각각의 노드들이 받게 되는 가중치를 계산할 수 있다는 내용이 된다.



자, 그럼 바로 PyTorch로 한번 간단하게 Backward propagation하여 가중치를 업데이트하는지 구현해보자.

## Module import


```python
import torch
from torch.autograd import Variable
```

## 변수 선언


```python
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = torch.tensor([1.0], requires_grad=True) # Any random
```

## 함수 선언

각 노드 x에 가중치 w를 반영할 forward(x)함수와 Loss function을 계산하기 위한 loss(x, y)함수를 선언한다.


```python
# our model forward pass
def forward(x):
    return x * w  # Loss function에 의해 l을 계산하고 이를 역전파하여 변수 w에 반영할 것임

# Loss function
def loss(y_pred, y_val):
    return (y_pred - y_val) ** 2
```

훈련 전 w는 업데이트 되지 않고 1인 것을 볼 수 있다.


```python
# Before training
print("Prediction (before training)",  4, forward(4).item())
```

    Prediction (before training) 4 4.0

10번의 epoch동안 gradient loss를 계산하고 이를 통해 가중치를 업데이트한다.

```python
# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward(x_val) # 1) Forward pass
        l = loss(y_pred, y_val) # 2) Compute loss
        l.backward() # 3) Back propagation to update weights
        print("\tgrad: ", x_val, y_val, w.grad.item())
        w.data = w.data - 0.01 * w.grad.item()

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print(f"Epoch: {epoch} | Loss: {l.item()}")
```

    	grad:  1.0 2.0 -2.0
    	grad:  2.0 4.0 -7.840000152587891
    	grad:  3.0 6.0 -16.228801727294922
    Epoch: 0 | Loss: 7.315943717956543
    	grad:  1.0 2.0 -1.478623867034912
    	grad:  2.0 4.0 -5.796205520629883
    	grad:  3.0 6.0 -11.998146057128906
    Epoch: 1 | Loss: 3.9987640380859375
    	grad:  1.0 2.0 -1.0931644439697266
    	grad:  2.0 4.0 -4.285204887390137
    	grad:  3.0 6.0 -8.870372772216797
    Epoch: 2 | Loss: 2.1856532096862793
    	grad:  1.0 2.0 -0.8081896305084229
    	grad:  2.0 4.0 -3.1681032180786133
    	grad:  3.0 6.0 -6.557973861694336
    Epoch: 3 | Loss: 1.1946394443511963
    	grad:  1.0 2.0 -0.5975041389465332
    	grad:  2.0 4.0 -2.3422164916992188
    	grad:  3.0 6.0 -4.848389625549316
    Epoch: 4 | Loss: 0.6529689431190491
    	grad:  1.0 2.0 -0.4417421817779541
    	grad:  2.0 4.0 -1.7316293716430664
    	grad:  3.0 6.0 -3.58447265625
    Epoch: 5 | Loss: 0.35690122842788696
    	grad:  1.0 2.0 -0.3265852928161621
    	grad:  2.0 4.0 -1.2802143096923828
    	grad:  3.0 6.0 -2.650045394897461
    Epoch: 6 | Loss: 0.195076122879982
    	grad:  1.0 2.0 -0.24144840240478516
    	grad:  2.0 4.0 -0.9464778900146484
    	grad:  3.0 6.0 -1.9592113494873047
    Epoch: 7 | Loss: 0.10662525147199631
    	grad:  1.0 2.0 -0.17850565910339355
    	grad:  2.0 4.0 -0.699742317199707
    	grad:  3.0 6.0 -1.4484672546386719
    Epoch: 8 | Loss: 0.0582793727517128
    	grad:  1.0 2.0 -0.1319713592529297
    	grad:  2.0 4.0 -0.5173273086547852
    	grad:  3.0 6.0 -1.070866584777832
    Epoch: 9 | Loss: 0.03185431286692619

훈련 후에 forward(x)함수에 4를 넣어보면 8에 가까운 숫자가 나오는 것을 확인할 수 있다.

```python
# After training
print("Prediction (after training)",  4, forward(4).item())
```

    Prediction (after training) 4 7.804864406585693

오늘 포스팅을 요약하면 PyTorch를 활용하면 계산하기 힘든 역전파와 Chain Rule을 손쉽게 구현할 수 있다는 것이다.

* 역전파는 l.backward()로 간단하게 구현할 수 있다.
* 가중치 업데이트는 w.data = w.data - 0.01 * w.grad.data로 구현할 수 있었다.

다음 포스팅은 PyTorch를 통해 하나의 신경망 모델을 처음부터 끝까지 구현해보는 것으로 구성해 보겠다.

도움이 되었길 바라며, 궁금한 점이 있다면 언제든 연락해도 편하게 들어오셔도 괜찮습니다.
