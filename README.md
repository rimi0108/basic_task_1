## 🛠 목적

기존 regression model로 구현된 MSE를 loss로 사용하고 있는 MNIST 실습 코드를 `nn.CrossEntropyLoss` 적용하는 것으로 바꾸기

MNIST는 0~9 숫자를 분류하는 문제니까,
해당 모델을 분류(classification) 모델로 바꿔야 함.

### 개념

**MSE (Mean Squared Error)**

- 회귀 (Regression)
- 목적 -> 실제 값과 예측 값의 차이 줄이기

**CrossEntropyLoss**

- 분류 (Classitication)
- 목적 -> 정답 클래스 확률을 높이기

같은 loss function이지만, 쓰는 문제 유형이 다름

**SoftMax**

- 여러 개의 숫자를 확률처럼 바꿔주는 함수
- 각 숫자를 0~1 사이로 바꾸고, 전체 합이 1이 되게 만들어줌

```python
logits = [2.0, 1.0, 0.1]

import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=0)
print(probs)

tensor([0.6590, 0.2424, 0.0986])
```

➡️ 확률 분포처럼 보여지는 것이 Softmax의 역할

- CrossEntropyLoss는 내부적으로 Softmax가 적용됨

**activation function(활성화 함수)**

- 뉴런(노드)가 출력값을 내보내기 전에 거치는 함수
- 활성화(activation) 할지를 결정한다.
- 자주 쓰이는 Activation -> ReLU, Sigmoid, Tanh, SoftMax

**은닉층, 출력층**

🧠

은닉층 = 두뇌에서 생각하고 조합하는 과정 (뇌가 다양하게 작동해야 하니까 ReLU 필요)

출력층 = 결론을 내리는 부분 (결과를 있는 그대로 보여줘야 하니까 손 안 대고 내보내는 게 중요)

**output dimension**

- 모델이 최종적으로 내보내는 출력의 모양(shape), 즉 Tensor의 크기 (dimension)을 말한다.

---

## 적용

전

```python
from torch import nn


class Model(nn.Module):
  def __init__(self, input_dim, n_dim):
    super().__init__()

    self.layer1 = nn.Linear(input_dim, n_dim)
    self.layer2 = nn.Linear(n_dim, n_dim)
    self.layer3 = nn.Linear(n_dim, 1)

    self.act = nn.ReLU()

  def forward(self, x):
    x = torch.flatten(x, start_dim=1)
    x = self.act(self.layer1(x))
    x = self.act(self.layer2(x))
    x = self.act(self.layer3(x))

    return x


model = Model(28 * 28 * 1, 1024)
```

후

```python
from torch import nn


class Model(nn.Module):
  def __init__(self, input_dim, n_dim, class_num):
    super().__init__()

    # 첫 번째 은닉층: input_dim 차원을 n_dim 차원으로 변환
    self.layer1 = nn.Linear(input_dim, n_dim)
    # 두 번째 은닉층: n_dim 차원의 은닉층을 다시 n_dim 차원으로 변환
    self.layer2 = nn.Linear(n_dim, n_dim)
   	# 세 번째 은닉층: n_dim 차원에서 클래스 수로 출력
    self.layer3 = nn.Linear(n_dim, class_num) # 🔄 클래스 수만큼!

    self.act = nn.ReLU()

  def forward(self, x):
    x = torch.flatten(x, start_dim=1)
    x = self.act(self.layer1(x))
    x = self.act(self.layer2(x))
    x = self.layer3(x) # 출력층 (layer3 에서 activation 뺌)

    # CrossEntropyLoss에서 softmax 안 해도 됨

    return x


model = Model(28 * 28 * 1, 1024, 10)
```

- 은닉층 (layer1, layer2)에서는 Activation 써야 하고, 출력층 (layer3) 에서는 빼야한다. (Why? `CrossEntropyLoss는 내부적으로 Softmax가 적용됨`)
- MNIST 데이터셋은 숫자 0~9까지, 총 10개 클래스를 분류하는 문제이다.
- 출력층에서 클래스 넘버만큼의 로짓(logits) 벡터를 반환해야 한다.
- CrossEntropyLoss 내부적으로 LogSoftmax + NLLLoss를 수행하므로, 모델 출력에는 softmax를 따로 쓰면 오히려 성능이 떨어짐
- forward 함수는 입력 데이터를 어떻게 처리할지 정의한다.

- input_dim -> 입력 차원
- n_dim -> 은닉층 차원
- class_num -> 클래스 개수
