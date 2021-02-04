 ---
 layout: post
 title:  "Dummy Post 78"
 date:   2021-02-05T14:25:52-05:00
 author: Ben Centra
 categories: Dummy
 ---

## t-test
- 사용 라이브러리 : scipy, numpy

### One-Sample T-test(단일 표본 t-검정)
- scipy의 stats 사용
- stats.ttest_1samp(데이터, 가설검정하고자하는 평균값) <- 이런식으로 사용함

#### (가설 설정)
- 전체 학생들 중 20명의 학생들을 추려 키를 재서 전체 학생들의 평균 키가 175cm인지 아닌지 알아보고 싶다.
- 귀무 가설 : 학생들의 평균 키가 175cm이다.
- 대립 가설 : 학생들의 평균 키가 175cm가 아니다.
<br><br>

### Unpaired T-test(독립 표본 t-검정)
- scipy.stats 의 ttest_ind 메소드를 이용
- scipy.stats.ttest_ind(표본1, 표본2, equal_var = False)
- equal_var은 등분산인 경우에 True / 등분산인지 잘 모르면 False로 두면됨
    - defualt = True로 되어있음

#### (가설 설정)
- 집단 1과 집단 2에서 각각 20명의 학생들을 추려, 각 집단의 키가 같은지, 다른지 알아보고 싶다.
- 귀무 가설: 두 집단의 평균 키는 같다.
- 대립 가설: 두 집단의 평균 키는 같지 않다.(양측 검정)
<br><br>

### Paired T-test(대응 표본 t-검정)
- scipy.stats 의 ttest_rel 메소드를 이용
- stats.ttest_rel(적용전데이터, 적용후데이터)

#### (가설검정)
- 다이어트 약을 복용한 사람들 중 20명을 추려 복용 전/후의 체중 차이가 유의미한지 알아보고 싶다.
- 귀무 가설: 복용 전/후의 체중 차이가 없다.
- 대립 가설: 복용 전/후의 체중 차이가 있다.


### 단측 검정은??
- 단측 검정은 양측 검정 값을 활용해서 계산할 수 있다.
- t-value < 0 and p-value / 2 < $\alpha$ 이면 -> less-than test의 귀무 가설을 기각
- t-value > 0 and p-value / 2 > $\alpha$ 이면 -> more-than test의 귀무 가설을 기각


### 관련 프로그래밍 지식
- np.random.normal(정규분포 평균, 표준편차, (행, 열) or 개수))
- print 관련 
    - %.3f : 소수점 3번째 자리까지 표현
    - %.i : integer 가져온다
    - '문자열 %.3f' % 소수를갖고있는 변수
    - 예시 : "The T-statistic is %.3f and the p-value is %.3f" % tTestResult


```python
import numpy as np
from scipy import stats

# def roundUp(num):
#     if (num - int(num)) >= 0.5 :
#         return int(num) + 1 
#     else :
#         return int(num)
```


```python
# to get consistent result
np.random.seed(1)

# generate 20 random heights with mean of 180, standard deviation of 5
heights = [180 + round(np.random.normal(0, 5), 1) for _ in range(20)]
print('data :', heights)

# perform 1-sample t-test
tTestResult = stats.ttest_1samp(heights, 175)
print('p-value :', tTestResult.pvalue)
print('t-value :', tTestResult.statistic)
```

    data : [188.1, 176.9, 177.4, 174.6, 184.3, 168.5, 188.7, 176.2, 181.6, 178.8, 187.3, 169.7, 178.4, 178.1, 185.7, 174.5, 179.1, 175.6, 180.2, 182.9]
    p-value : 0.0027778735992021776
    t-value : 3.4346467603023054



```python
# print result
print("The T-statistic is %.3f and the p-value is %.3f" % tTestResult)
```

    The T-statistic is 3.435 and the p-value is 0.003



```python
[np.random.normal(0, 5) for _  in range(20)]
```




    [-3.7719897049832642,
     6.264340776166439,
     2.564649102090044,
     -1.4904641755135783,
     2.442590732687485,
     -0.3778585651052786,
     5.6581469372571345,
     7.599084082110994,
     10.927877032665807,
     -6.982481677440688,
     -7.220569027147947,
     -2.5223293147322563,
     0.8001853472391524,
     4.3808446055811245,
     1.5781747362080263,
     -10.111006079120015,
     -1.5310200631418591,
     4.139873213036231,
     1.150473676821917,
     3.8100559015601236]




```python

```


```python
#to get consistent result
np.random.seed(1)
 
# group 1 heights : mean 170, standard deviation 5
group1Heights = [170 + np.random.normal(0, 5) for _ in range(20)]
# group 2 heights : mean 180, standard deviation 10
group2Heights = [175 + np.random.normal(0, 10) for _ in range(20)]
 
# perform t-test assuming equal variances
tTestResult = stats.ttest_ind(group1Heights, group2Heights,  equal_var=True)
 
# perform t-test NOT assuming equal variances
tTestResultDiffVar = stats.ttest_ind(group1Heights, group2Heights, equal_var=False)
```


```python
print("The t-statistic and p-value assuming equal variances is %.3f and %.3f." % tTestResult)
print("The t-statistic and p-value not assuming equal variances is %.3f and %.3f" % tTestResultDiffVar)
```

    The t-statistic and p-value assuming equal variances is -2.329 and 0.025.
    The t-statistic and p-value not assuming equal variances is -2.329 and 0.026



```python

```


```python
#to get consistent result
np.random.seed(1)
 
#before treatment : mean 60, standard deviation 5
beforeWeights = [60 + np.random.normal(0, 5) for _ in range(20)]
#after treatment : mean 0.99-fold decrease, standard deviation 0.02
afterWeights = [w * np.random.normal(0.99, 0.02) for w in beforeWeights]
 
#perform paired t-test
tTestResult = stats.ttest_rel(beforeWeights, afterWeights)
```


```python
print("The T-statistic is %.3f and the p-value is %.3f" % tTestResult)
```

    The T-statistic is 2.915 and the p-value is 0.009

