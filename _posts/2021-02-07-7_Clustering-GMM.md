---
layout: post
title: GMM
date: 2021-02-07T01:15:37
author: SCENE
categories: DATA
---
## Clustering_GMM
- GMM 성능 평가 : [링크](https://www.researchgate.net/post/What_is_the_best_criteria_for_GMM_model_selection)
- GMM 성능 평가 이론 : [링크](https://towardsdatascience.com/gaussian-mixture-model-clusterization-how-to-select-the-number-of-components-clusters-553bef45f6e4)


```python
# shift + L하면 code line표시됨

# 라이브러리
import pandas as pd
import sklearn as sk
import numpy as np
# import random

# scaling을 하기 위한 라이브러리
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

# pca 라이브러리
from sklearn.decomposition import PCA

# 클러스터링 위한 라이브러리
# from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# 시각화를 위한 세팅
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [12, 8]
```

## 1. Data 불러오기
- 데이터 출처 : https://www.kaggle.com/arjunbhasin2013/ccdata#


```python
df = pd.read_csv('data/credit_card.csv', header = 0, )
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUST_ID</th>
      <th>BALANCE</th>
      <th>BALANCE_FREQUENCY</th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>CASH_ADVANCE</th>
      <th>PURCHASES_FREQUENCY</th>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <th>CASH_ADVANCE_TRX</th>
      <th>PURCHASES_TRX</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
      <th>MINIMUM_PAYMENTS</th>
      <th>PRC_FULL_PAYMENT</th>
      <th>TENURE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C10001</td>
      <td>40.900749</td>
      <td>0.818182</td>
      <td>95.40</td>
      <td>0.00</td>
      <td>95.4</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>2</td>
      <td>1000.0</td>
      <td>201.802084</td>
      <td>139.509787</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C10002</td>
      <td>3202.467416</td>
      <td>0.909091</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>6442.945483</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>4</td>
      <td>0</td>
      <td>7000.0</td>
      <td>4103.032597</td>
      <td>1072.340217</td>
      <td>0.222222</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C10003</td>
      <td>2495.148862</td>
      <td>1.000000</td>
      <td>773.17</td>
      <td>773.17</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>12</td>
      <td>7500.0</td>
      <td>622.066742</td>
      <td>627.284787</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C10004</td>
      <td>1666.670542</td>
      <td>0.636364</td>
      <td>1499.00</td>
      <td>1499.00</td>
      <td>0.0</td>
      <td>205.788017</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>1</td>
      <td>1</td>
      <td>7500.0</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C10005</td>
      <td>817.714335</td>
      <td>1.000000</td>
      <td>16.00</td>
      <td>16.00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>1200.0</td>
      <td>678.334763</td>
      <td>244.791237</td>
      <td>0.000000</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 전반적인 데이터 살펴보기
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BALANCE</th>
      <th>BALANCE_FREQUENCY</th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>CASH_ADVANCE</th>
      <th>PURCHASES_FREQUENCY</th>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <th>CASH_ADVANCE_TRX</th>
      <th>PURCHASES_TRX</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
      <th>MINIMUM_PAYMENTS</th>
      <th>PRC_FULL_PAYMENT</th>
      <th>TENURE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8950.000000</td>
      <td>8950.000000</td>
      <td>8950.000000</td>
      <td>8950.000000</td>
      <td>8950.000000</td>
      <td>8950.000000</td>
      <td>8950.000000</td>
      <td>8950.000000</td>
      <td>8950.000000</td>
      <td>8950.000000</td>
      <td>8950.000000</td>
      <td>8950.000000</td>
      <td>8949.000000</td>
      <td>8950.000000</td>
      <td>8637.000000</td>
      <td>8950.000000</td>
      <td>8950.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1564.474828</td>
      <td>0.877271</td>
      <td>1003.204834</td>
      <td>592.437371</td>
      <td>411.067645</td>
      <td>978.871112</td>
      <td>0.490351</td>
      <td>0.202458</td>
      <td>0.364437</td>
      <td>0.135144</td>
      <td>3.248827</td>
      <td>14.709832</td>
      <td>4494.449450</td>
      <td>1733.143852</td>
      <td>864.206542</td>
      <td>0.153715</td>
      <td>11.517318</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2081.531879</td>
      <td>0.236904</td>
      <td>2136.634782</td>
      <td>1659.887917</td>
      <td>904.338115</td>
      <td>2097.163877</td>
      <td>0.401371</td>
      <td>0.298336</td>
      <td>0.397448</td>
      <td>0.200121</td>
      <td>6.824647</td>
      <td>24.857649</td>
      <td>3638.815725</td>
      <td>2895.063757</td>
      <td>2372.446607</td>
      <td>0.292499</td>
      <td>1.338331</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>50.000000</td>
      <td>0.000000</td>
      <td>0.019163</td>
      <td>0.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>128.281915</td>
      <td>0.888889</td>
      <td>39.635000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1600.000000</td>
      <td>383.276166</td>
      <td>169.123707</td>
      <td>0.000000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>873.385231</td>
      <td>1.000000</td>
      <td>361.280000</td>
      <td>38.000000</td>
      <td>89.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.083333</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>3000.000000</td>
      <td>856.901546</td>
      <td>312.343947</td>
      <td>0.000000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2054.140036</td>
      <td>1.000000</td>
      <td>1110.130000</td>
      <td>577.405000</td>
      <td>468.637500</td>
      <td>1113.821139</td>
      <td>0.916667</td>
      <td>0.300000</td>
      <td>0.750000</td>
      <td>0.222222</td>
      <td>4.000000</td>
      <td>17.000000</td>
      <td>6500.000000</td>
      <td>1901.134317</td>
      <td>825.485459</td>
      <td>0.142857</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>19043.138560</td>
      <td>1.000000</td>
      <td>49039.570000</td>
      <td>40761.250000</td>
      <td>22500.000000</td>
      <td>47137.211760</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.500000</td>
      <td>123.000000</td>
      <td>358.000000</td>
      <td>30000.000000</td>
      <td>50721.483360</td>
      <td>76406.207520</td>
      <td>1.000000</td>
      <td>12.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.columns[df.isnull().any(axis=0)]].isnull().sum()
```




    CREDIT_LIMIT          1
    MINIMUM_PAYMENTS    313
    dtype: int64




```python
# 여기서는 CREDIT_LIMIT 에 1개의 null값이 있음
# 따라서, CREDIT_LIMIT값이 null인 row 살펴봄
df[df['CREDIT_LIMIT'].isnull()==True]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUST_ID</th>
      <th>BALANCE</th>
      <th>BALANCE_FREQUENCY</th>
      <th>PURCHASES</th>
      <th>ONEOFF_PURCHASES</th>
      <th>INSTALLMENTS_PURCHASES</th>
      <th>CASH_ADVANCE</th>
      <th>PURCHASES_FREQUENCY</th>
      <th>ONEOFF_PURCHASES_FREQUENCY</th>
      <th>PURCHASES_INSTALLMENTS_FREQUENCY</th>
      <th>CASH_ADVANCE_FREQUENCY</th>
      <th>CASH_ADVANCE_TRX</th>
      <th>PURCHASES_TRX</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
      <th>MINIMUM_PAYMENTS</th>
      <th>PRC_FULL_PAYMENT</th>
      <th>TENURE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5203</th>
      <td>C15349</td>
      <td>18.400472</td>
      <td>0.166667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>186.853063</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.166667</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>9.040017</td>
      <td>14.418723</td>
      <td>0.0</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 1개 밖에 없기 때문에 전체 데이터 트렌드에 영향을 줄 가능성이 적기 때문에
# 제거하고자함 -> 제거 안하면 PCA 등 모델링 하는 과정에서 문제가 생길 수 있음
# df.drop 썻을땐 index reset 안해줘도됨
# 8950 -> 8949 로 1개 준 것을 알 수 있음
print(len(df))
row_idx = list(df[df['CREDIT_LIMIT'].isnull()==True].index)
df = df.drop(row_idx)
print(len(df))
```

    8950
    8949



```python
# 이부분은 몇개의 column만 사용하기위해 칼럼 추출하는 과정
# 칼럼 선택은 임의로 하였음
df = df[['CUST_ID', 'BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUST_ID</th>
      <th>BALANCE</th>
      <th>PURCHASES</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C10001</td>
      <td>40.900749</td>
      <td>95.40</td>
      <td>1000.0</td>
      <td>201.802084</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C10002</td>
      <td>3202.467416</td>
      <td>0.00</td>
      <td>7000.0</td>
      <td>4103.032597</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C10003</td>
      <td>2495.148862</td>
      <td>773.17</td>
      <td>7500.0</td>
      <td>622.066742</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C10004</td>
      <td>1666.670542</td>
      <td>1499.00</td>
      <td>7500.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C10005</td>
      <td>817.714335</td>
      <td>16.00</td>
      <td>1200.0</td>
      <td>678.334763</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Scaling 하기 
__scikit-learn에서는 다음과 같은 스케일링 클래스를 제공한다.__
- StandardScaler(X) : 평균이 0과 표준편차가 1이 되도록 변환. 
    - 계산식 : z = (x - u) / s
- MinMaxScaler(X) : 최대값이 각각 1, 최소값이 0이 되도록 변환
- MaxAbsScaler(X) : 0을 기준으로 절대값이 가장 큰 수가 1또는 -1이 되도록 변환
- RobustScaler(X) : 중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환. 
    - 아웃라이어의 영향을 최소화한 기법. 즉, 4개 중에 가장 이상치의 영향이 적음

__출처__
- https://mkjjo.github.io/python/2019/01/10/scaler.html
- https://datascienceschool.net/view-notebook/f43be7d6515b48c0beb909826993c856/


```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
```


```python
## column 단위로 scaling 하는 과정에서 fit_transform 및 reshape과정이 필요
for i in range(1, len(df.columns)):
#     df.iloc[:,i] = StandardScaler().fit_transform(df.iloc[:,i].values.reshape(-1,1))
#     df.iloc[:,i] = MinMaxScaler().fit_transform(df.iloc[:,i].values.reshape(-1,1))
#     df.iloc[:,i] = MaxAbsScaler().fit_transform(df.iloc[:,i].values.reshape(-1,1))
    df.iloc[:,i] = RobustScaler().fit_transform(df.iloc[:,i].values.reshape(-1,1))
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CUST_ID</th>
      <th>BALANCE</th>
      <th>PURCHASES</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C10001</td>
      <td>-0.432387</td>
      <td>-0.248596</td>
      <td>-0.408163</td>
      <td>-0.431661</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C10002</td>
      <td>1.209127</td>
      <td>-0.337724</td>
      <td>0.816327</td>
      <td>2.138325</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C10003</td>
      <td>0.841881</td>
      <td>0.384615</td>
      <td>0.918367</td>
      <td>-0.154807</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C10004</td>
      <td>0.411728</td>
      <td>1.062726</td>
      <td>0.918367</td>
      <td>-0.564601</td>
    </tr>
    <tr>
      <th>4</th>
      <td>C10005</td>
      <td>-0.029058</td>
      <td>-0.322776</td>
      <td>-0.367347</td>
      <td>-0.117739</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BALANCE</th>
      <th>PURCHASES</th>
      <th>CREDIT_LIMIT</th>
      <th>PAYMENTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8949.000000</td>
      <td>8949.000000</td>
      <td>8949.000000</td>
      <td>8949.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.358756</td>
      <td>0.599631</td>
      <td>0.304990</td>
      <td>0.577257</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.080777</td>
      <td>1.996252</td>
      <td>0.742615</td>
      <td>1.907230</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.453623</td>
      <td>-0.337724</td>
      <td>-0.602041</td>
      <td>-0.564601</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.386974</td>
      <td>-0.300541</td>
      <td>-0.285714</td>
      <td>-0.312109</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.613026</td>
      <td>0.699459</td>
      <td>0.714286</td>
      <td>0.687891</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.433744</td>
      <td>45.477807</td>
      <td>5.510204</td>
      <td>32.848838</td>
    </tr>
  </tbody>
</table>
</div>



### 3. PCA로 차원 축소하기
__(PCA를 사용함으로서 얻을 수 있는 장점)__
- 기존 변수를 조합해 새로운 변수를 만드는 변수 추출(Feature Extraction)기법
- 1)차원의 저주 문제 : 차원을 축소해주기 때문
- 2)다중공선성 문제 : CA 알고리즘은 주성분 PC1과 PC2를 찾는 과정에서 두 변수가 직교해야하기때문에 두 변수 사이의 상관관계가 0으로 나타나 다중공선성 문제를 해결


```python
# cloumn_cnt를 가져오는 이유는 마지막 column까지 index로 선택해주기 위함
column_cnt = len(list(df))
print(column_cnt, '
')

# n_components는 몇개의 주성분을 추출할지에 대한 정보를 받는 parameter
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df.iloc[:,1:column_cnt])
print(principalComponents, '
')

pca_df = pd.DataFrame(data = principalComponents, columns = ['pca1', 'pca2'])
print(pca_df.head())
```

    5 
    
    [[-1.50397504 -0.46367594]
     [ 0.58569861  1.98480511]
     [-0.4819072   0.01798893]
     ...
     [-1.52614805 -0.54483616]
     [-1.65031872 -0.48325463]
     [-0.87080777 -1.06752393]] 
    
           pca1      pca2
    0 -1.503975 -0.463676
    1  0.585699  1.984805
    2 -0.481907  0.017989
    3 -0.340593 -0.860348
    4 -1.278694 -0.053911



```python
# n_components : The number of mixture components. => 몇개의 집단으로 나눌건지에 대한거라고 생각하면 됨
gmm = GaussianMixture(n_components=2, random_state = 33)
gmm.fit(pca_df)

# 모델 바탕으로 예측
pca_df['predict'] = gmm.predict(pca_df)

# GMM모델은 center가 없음
# C = gmm.cluster_centers_
# print('center point of cluster :', C)
```


```python
pca_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pca1</th>
      <th>pca2</th>
      <th>predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.503975</td>
      <td>-0.463676</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.585699</td>
      <td>1.984805</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.481907</td>
      <td>0.017989</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.340593</td>
      <td>-0.860348</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.278694</td>
      <td>-0.053911</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot('pca1', 'pca2',  data = pca_df,
         linestyle='none', 
         marker='o', 
         markersize=10,
         color='blue', 
         alpha=0.5)
plt.title('example', fontsize=20)
plt.xlabel('X_axis', fontsize=14)
plt.ylabel('Y_axis', fontsize=14)
plt.show()
```


    
![png](/images/7_Clustering-GMM_files/7_Clustering-GMM_18_0.png)
    



```python
groups = pca_df.groupby('predict')
print(list(groups))
```

    [(0,           pca1      pca2  predict
    0    -1.503975 -0.463676        0
    2    -0.481907  0.017989        0
    3    -0.340593 -0.860348        0
    4    -1.278694 -0.053911        0
    5     0.011052 -0.370625        0
    ...        ...       ...      ...
    8944 -1.320265 -0.542146        0
    8945 -1.337105 -0.568938        0
    8946 -1.526148 -0.544836        0
    8947 -1.650319 -0.483255        0
    8948 -0.870808 -1.067524        0
    
    [6751 rows x 3 columns]), (1,           pca1      pca2  predict
    1     0.585699  1.984805        1
    6     6.278525 -1.971382        1
    12    0.924391 -1.889339        1
    13    0.757344 -0.788102        1
    15    1.042460  0.992355        1
    ...        ...       ...      ...
    8763  0.889408 -0.876861        1
    8834  0.866200  0.385985        1
    8856  3.275819  2.621334        1
    8896  1.030184 -1.490055        1
    8913  0.403814  1.340784        1
    
    [2198 rows x 3 columns])]



```python
# group 끼리 묶어줌
groups = pca_df.groupby('predict')

fig, ax = plt.subplots()
for groupidx, data in groups:
    ax.plot(data.pca1, data.pca2, 
            marker='o', 
            linestyle='',
            label=groupidx)
ax.legend(fontsize=12)
```




    <matplotlib.legend.Legend at 0x7ff315f1e6d0>




    
![png](/images/7_Clustering-GMM_files/7_Clustering-GMM_20_1.png)
    
