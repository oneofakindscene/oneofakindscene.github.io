---
layout: post
title: MCMC(Markov Chain Monte Carlo)
date: 2021-02-14T12:11:00
author: Scene
categories: [Data Science]
use_math: true
---

## __MCMC(Markov Chain Monte Carlo)__
<br>

(Reference)
- [참고링크 : 공돌이의 수학정리노트](https://angeloyeo.github.io/2020/09/17/MCMC.html)<br>
- [참고링크 : 잠재 디리클레 할당 파헤치기 1~3탄](https://bab2min.tistory.com/567?category=673750)

(용어 정리)
- 타켓 분포 : 우리가 샘플을 추출하고자 하는 유사 확률 분포 $\Rightarrow$ $f(x)$ 로 표기함
- 제안 분포 : 제안 분포 는 우리가 쉽게 샘플을 추출할 수 있는 분포를 이용해서 생성할 수 있다. 가령, uniform distribution이 생긴게 가장 단순하기 때문에 uniform distribution을 이용해서 제안 분포를 만들 수 있음 $\Rightarrow$ $g(x)$ 로 표기함

### __MCMC의 정의__
- 마르코프 연쇄의 구성에 기반한 확률 분포로부터 원하는 분포의 정적 분포를 갖는 __표본을 추출하는 알고리즘__ 의 한 종류이다
- 즉, __MCMC는 샘플링 방법 중 하나__
- MCMC는 Monte Carlo와 Markov Chain의 개념을 합친 것
- MCMC에서는 “통계적인 특성을 이용해 무수히 뭔가를 많이 시도해본다”는 의미에서 Monte Carlo라는 이름을 붙였다고 보면 좋을 것 같다.
- MCMC는 샘플링 방법 중 하나라고 하였는데, “가장 마지막에 뽑힌 샘플이 다음번 샘플을 추천해준다”는 의미에서 Markov Chain이라는 단어가 들어갔다고 보면 좋을 것 같다.
- MCMC를 수행한다는 것은 첫 샘플을 랜덤하게 선정한 뒤, 첫 샘플에 의해 그 다음번 샘플이 추천되는 방식의 시도를 무수하게 해본다는 의미를 갖고 있다.
- 활용 분야
  - sampling $\Rightarrow$ 타겟분포와 제안분포를 활용해서 타겟분포와 유사한 sample을 추출해줌
  - parameter estimation(파라미터 추정) $\Rightarrow$ random initialization에서 뽑힌 임의 값을 기반으로 parameter(평균, 표준폍차 등)를 찾아가는 과정

### __표본을 추출하는 알고리즘 MCMC의 종류__
- [Metropolis](https://angeloyeo.github.io/2020/09/17/MCMC.html) : Metropolis는 __symmetric__ 한 확률분포를 사용하는 경우에 대한 알고리즘을 제안
  - 수식
    - $f(x)$ : 타겟분포
    - $g(x)$ : 제안분포
    - ${f(x_1) \over f(x_0)} > 1$인 경우 accept
    - 위를 만족하지 못할 경우, ${f(x_1) \over f(x_0)} > u$ 인 경우 accept
    - $u$ 값은 uniform 분포 $U_{(0,1)}$ 에서 임의로 추출한 값
    - 제안 분포 $g(x)$ 의 역할은? : 제안 분포 내에서 다음 포인트($x_1$)를 추천 받음
- [Metropolis-Hastings](https://angeloyeo.github.io/2020/09/17/MCMC.html) : symmetric 하지 않더라도 일반적인 확률 분포에 대한 경우까지 어떻게 수학적으로 계산할 수 있는지에 관해 기존 Metropolis 알고리즘을 개선한 알고리즘

- Gibbs Sampling :

---

### __참고__

#### Monte Carlo
- Monte Carlo는 쉽게 말해 통계적인 수치를 얻기 위해 수행하는 __‘시뮬레이션’__ 같은 것 $\Rightarrow$ __'시뮬레이션'__ 의 목적은 유한한 시도만으로 정답을 추정하기 위함
- MCMC를 수행한다는 것은 첫 샘플을 랜덤하게 선정한 뒤, 첫 샘플에 의해 그 다음번 샘플이 추천되는 방식의 시도를 무수하게 해본다는 의미를 갖고 있다.
- 위의 문장에서 가장 수학적이지 않은 단어는 뭘까? 바로 “추천”이다.
- 또한, 추천 다음에는 “승낙/거절”의 단계까지도 포함됨
- 이 post에서 주요하게 소개하는 MCMC 샘플링 알고리즘은 Metropolis 알고리즘

#### Markov Chain
- Markov Chain은 어떤 상태에서 다음 상태로 넘어갈 때, __바로 전 단계의 상태에만 영향을 받는 확률 과정을 의미__ 한다.
  - 보통 사람들은 전날 먹은 식사와 유사한 식사를 하지 않으려는 경향이 있다.
  - `가령, 어제 짜장면을 먹었다면 오늘은 면종류의 음식을 먹으려고 하지 않는다`
  - 이렇듯 음식 선택이라는 확률과정에 대해 오늘의 음식 선택이라는 과정은 어제의 음식 선택에만 영향을 받고, 그저께의 음식 선택에는 영향을 받지 않는다면 이 과정은 마르코프 성질(Markov property)을 가진다고 할 수 있으며, 이 확률 과정은 Markov chain이라고 할 수 있다.
- MCMC는 샘플링 방법 중 하나라고 하였는데, “가장 마지막에 뽑힌 샘플이 다음번 샘플을 추천해준다”는 의미에서 Markov Chain이라는 단어가 들어갔다고 보면 좋을 것 같다.

#### 샘플링 과정 (Metropolis 알고리즘)
- 우선은 Rejection Sampling을 할 때와 마찬가지로 샘플을 추출하고자 하는 __target 분포__ 가 하나 있어야 한다.
1. random initialization
    - 쉽게 말해 샘플 공간에서 아무런 입력값이나 하나를 선택해 주는 것
2. 제안 분포로부터 다음 포인트를 추천받기


#### Stochastic Process(=Random Process)
- Stochastic Process는 Random Process 라고도 함 $\Rightarrow$ 전자는 수학쪽에서 후자는 공대쪽에서 많이 사용되는 용어
- Stochastic Process(=Random Process)는 시간별로 표시된(indexed by time) 확률변수의 집합(또는 모음) $\Rightarrow$ 즉, 시간에 따라 확률도 변하게 된다는 뜻 입니다.
- 예를 들어, 주가도 시간에 따라 연속적으로 변함 $\Rightarrow$ 이것을 연속시간 랜덤과정(continuous-time random process)이라고 합니다.

#### parameter 추정시에 활용하는 ${f_{new}(x) \over f_{old}(x)} > 1$의 의미
- 수식의 의미 : x값은 동일하며, 이전 파라미터(예를 들어 평균)에 비해 지금 파라미터가 더 좋은가를 확인
- ${f_{old}(x)}$ : 평균이 old일때의 분포에서 x일때의 제안분포에서의 값
- ${f_{new}(x)}$ : 평균이 new일때의 분포에서 x일때의 제안분포에서의 값

#### LDA의 어떤점을 차용??
- Bayseian network의 가장 상류에 Multinomial 분포 추가
- Multinomial(다항분포)
  - K 개의 가능성 중 하나를 선택하는 문제
  - 다항 분포에서 차원이 2인 경우 이항 분포가 된다.
- 문서의 토픽분포 및 토픽별 단어분포를 결합하는 부분을 차용한 건가??
- 주제가 k가지가 있다고 하면 k개의 주제 중 하나를 고르는 행위는 다항분포가 될겁니다. 또한 주제에 포함된 단어가 v개가 있다고 할때 v개의 단어 중 하나를 고르는 행위 역시 다항분포가 되겠죠
- 우리가 관측하는 단어는 다항분포의 결과가 두번 중첩된 것이죠. 따라서 사전 확률은 켤레 사전 분포인 디리클레 분포로 두면 되겠습니다
- 그렇기에 문헌별 주제분포와 주제별 단어분포는 디리클레 분포를 따른다고 가정합니다.
- 단어(W)를 한개한개 관측해나가면서 단어마다 적절한 주제를 부여하여 해당 단어가 속한주제의 번호(Z)를 정해줌 $\Rightarrow$ 그리고 이 결과를 따라 문서의 주제 분포, 주제의 단어 분포를 업데이트 하는 과정을 거침
- 이러한 과정을 통해서 가능한 모든 경우 Z값 중에서 가장 가능도가 높은 Z값을 찾아내게 되면, 문헌 내의 각각의 단어들이 어디에 배정되어야하는지 추론할 수 있게되는거죠.
- 디리클레-다항 분포는 말 그대로 디리클레 분포와 다항 분포를 합친 것이죠. 디리클레 분포에서 특정 확률 분포를 뽑아내고, 이 확률 분포를 여러 번 시행하여 최종적인 분포를 뽑는 과정

#### 켤레 분포
- 켤레 분포란? : 즉, 사전분포(Prior distribution)와 사후분포(Posterior distribution)가 동일한 분포족에 속하면 사전분포를 켤레사전분포라고 한다.
- 디리클레 분포는 다항 분포를 일반화한 것
