# ✈️ AirlineSent  
![header](https://capsule-render.vercel.app/api?type=waving&color=4B89DC&height=300&section=header&text=AirlineSent&fontSize=80&animation=fadeIn&fontAlignY=36&descSize=25&desc=Sentiment%20Analysis%20and%20Evaluation%20of%20Airline%20Reviews&descAlignY=53&fontColor=FFFFFF)

> MobileBERT를 활용한 항공사 리뷰 감성 분류 및 종합 분석  
> 항공사별 평점, 긍정률, 일관성 기반 종합 평가

---

## 1. 개요

### ✈️ 프로젝트 목적

전자상거래와 마찬가지로 항공산업에서도 사용자 리뷰는 항공사 선택에 큰 영향을 준다.  
이 프로젝트는 실제 항공사 리뷰 데이터를 기반으로 리뷰를 감성(긍정/부정)으로 분류하고,  
**연도별 성과, 종합 평가**까지 시각화하여  
항공사별 고객 만족도를 직관적으로 파악할 수 있도록 한다.

---
## 2. 모델 구성

### 🔍 MobileBERT 활용

| 항목 | 내용 |
|------|------|
| 모델 | `google/mobilebert-uncased` |
| 목적 | 감성 분류 (긍/부정) |
| 데이터 | 항공사 리뷰 5,000건 |
| 에포크 | 10 |
| 평가 | Accuracy, F1-score |

---
## 3. 데이터

### 📂 데이터 출처
- 본 프로젝트 이번 Kaggle에서 제공하는 Airline Reviews 데이터셋을 사용
- Kaggle  https://www.kaggle.com/datasets/juhibhojani/airline-reviews
- 주요 컬럼:  
  `AirlineName`, `OverallScore`, `DateFlown`, `Review`, `SeatType`, `CabinStaff`, `Food`, ...
#### [ 데이터 구성 ]

| AirlineName       | OverallScore | Review 내용         | DateFlown (탑승일) | DatePub (작성일) | TypeOfTraveler (여행자 유형) | SeatType (좌석등급) | CabinStaff (승무원 서비스) | Food (기내식) | Entertainment (엔터테인먼트) |
|-------------------|--------------|----------------------|--------------------|------------------|-------------------------------|---------------------|-----------------------------|---------------|-------------------------------|
| 항공사 이름       | 전반적인 평점| 탑승객이 작성한 리뷰 | 항공기 실제 탑승 날짜| 리뷰가 등록된 날짜 | 비즈니스/휴가 등 여행자 분류 | Economy/Business 등 | 승무원 친절도 등 평가       | 기내식 만족도  | 영화, 음악 등 오락서비스 평가  |

#### [ 데이터 예시 ]

| index | AirlineName       | OverallScore | Review                                                                 | DateFlown | DatePub   | TypeOfTraveler | SeatType  | CabinStaff | Food | Entertainment |
|-------|--------------------|--------------|------------------------------------------------------------------------|-----------|-----------|----------------|-----------|------------|------|----------------|
| 0     | Qatar Airways      | 10           | Outstanding service and comfortable seats.                            | 2023-05   | 2023-06-01| Business        | Business  | 10         | 9    | 10             |
| 1     | Singapore Airlines | 9            | Good flight experience but the food could be better.                  | 2023-04   | 2023-05-15| Economy         | Economy   | 9          | 7    | 9              |
| 2     | Emirates           | 8            | The staff were friendly, and the in-flight entertainment was amazing. | 2023-03   | 2023-04-20| Economy         | Economy   | 8          | 8    | 10             |
| ...   | ...                | ...          | ...                                                                    | ...       | ...       | ...            | ...       | ...        | ...  | ...            |

### 🧹 전처리

- `OverallScore` 기준으로 감성 라벨 생성  
  - 1~2점: 부정  
  - 9~10점: 긍정
- 한글 및 특수문자 포함 리뷰 제거
- 학습용 샘플 5,000건 랜덤 추출

### 📈 리뷰 점수 분포
> 리뷰 대부분이 긍정적 (7~10점)으로 분포
 - <img src="review.png">

 * ### 분석 데이터
       
 - 분석을 위해 평점이 9~10인 리뷰에는 레이블 1을 평점이 1~2 인 리뷰에는 레이블 0을 부여하고 나머지 평점들은 전부 삭제한 후,
   'Review text','label' 데이터로 새로운 데이터셋을 만들었다.

      | index | Review text                                      | label |
      |-------|--------------------------------------------------|-------|
      | 1     | Holguin to Havana last week. Okay apart from ... | 0     |A great read to continue the Bride Train serie...| 
      | ...   | ...                                              | ...   
      | 20841 | Rome to Prague. Was very happy with the ...      | 1     |0|
     
       

  * ### 데이터셋 분할 - 학습 데이터 생성

      학습 데이터를 만들기 위해 5000개를 랜덤 추출하였다. 

      | index | ReviewText                                           |label|
      |------|------------------------------------------------------|-|
      | 1    | Considering the overall flying time from Muscat ...  |1|
      | 2    | First experience flying Small Planet was a great ... |1|
      | ...  | ...                                                  |...|
      | 4999 | This was one of my worst airline experiences ...     |0|
      | 5000 | Las Palmas to Nurnberg. We flew Corendon for ...     |0|

<hr>
---

## 4. 시각화 및 분석


### 📊 항공사별 리뷰 수
> 가장 많은 리뷰가 등록된 항공사 Top 10 시각화
 - 현재 데이터셋에는 총 515개의 항공사가 포함되어 있어 모든 항공사를 한 번에 시각화하기엔 가독성이 떨어집니다.
따라서 가장 많은 리뷰 수를 가진 상위 10개 항공사를 기준으로 분포를 시각화하여 전체적인 흐름을 파악했습니다.
 - <img src="airline.png">

### 🏆 연도별 최고 항공사

- 주피터랩으로 시각화시 가독성이 떨어져 표를 만들어 시각화 해봤습니다.

| 연도   | 항공사 | 평균 평점 |
|------|--------|-------|
| 2024 | Qatar Airways | 10.0  |
| 2023 | Qatar Airways | 9.6   |
| 2022 | Qatar Airways | 9.3   |
| 2021 | Emirates | 9.3   |
| 2020 | Qatar Airways | 9.6   |
| 2019 | Singapore Airlines | 9.3   |
| ...  | ...    | ...   |
| 2013 | 	Asiana Airlines | 10.0  |
| 2012 | 	Asiana Airlines | 10.0  |
| 2011 | 	Qatar Airways | 10.0  |
| 2010 | 	Qatar Airways | 9.6   |
| 2009 | 	Etihad Airways | 10.0   |
| 2008 | 	Air New Zealand | 10.0    |
- ✈️ Qatar Airways가 여러 해에 걸쳐 최고의 평점을 차지하며 고객 만족도를 유지하고 있습니다.
### 🧠 종합 평가: Top 5 항공사

항공사별로 다음 4가지 항목을 기준으로 점수를 부여

| 기준 | 설명 |
|------|------|
| 리뷰 수 | 전체 리뷰 개수 |
| 평균 평점 | OverallScore 평균 |
| 긍정 비율 | 8점 이상 리뷰 비중 |
| 일관성 | 평점의 표준편차 역수 (낮을수록 좋음) |

#### 📊 Top 5 종합 항공사

| 순위 | 항공사 | 종합 점수 |
|------|--------|------------|
| 1 | Singapore Airlines | 3.85 |
| 2 | Emirates | 3.76 |
| 3 | Qatar Airways | 3.72 |
| 4 | ANA | 3.69 |
| 5 | Lufthansa | 3.61 |

---

## 5. 결과 및 결론

 * ### MobileBERT를 사용하여 학습한 결과

 <img src="model.png">


 - 각 단계의 loss와 Accuracy의 평균을 내어 나타내보면 아래와 같다. 

    |step| 7      | 8     | 9      | 10     |
    |-|--------|-------|--------|--------|
    |loss| 96,990 | 0.37  | 0.2149 | 0.1567 |
    |Accuracy| 0.877  | 0.911 | 0.9167 | 0.9238 |

 모델은 총 10 에폭(epoch)에 걸쳐 학습되었으며, 학습 정확도(Train Accuracy)와 검증 정확도(Validation Accuracy) 모두 높은 수준에서 수렴하는 모습을 보였다.
- 최종 Train Accuracy: 85.9%

- 최종 Validation Accuracy: 85.3%

- Train/Validation 간 차이: 약 0.6% 이내로, 과적합(overfitting) 없이 일반화 성능이 매우 우수함을 확인할 수 있다.

 또한, 손실 값(Loss)의 절댓값은 비교적 높게 나타났지만 이는 모델 구조(MobileBERT), 데이터 규모, 손실 함수 등 다양한 요인에 따라 나타나는 자연스러운 현상이다. 중요한 것은 손실 값이 학습 과정 중 점차 감소하고 있고, 정확도 지표에서 매우 안정적인 학습이 이루어졌다는 점이다.
    
 * ### 분석 데이터 전체에 적용한 결과값

    <img src = "inference.png">

 - 결과 요약
   - 정확도(Accuracy): 0.8557 → 85.6%
   - Macro F1-score: 0.84 → 클래스별 균형 잘 잡힘 
   - Weighted F1-score: 0.86 → 실제 분포를 반영한 가중 평균도 높음

   
### 결론
 - MobileBERT로 학습한 모델은 약 **0.85 이상의 정확도**로 항공사 리뷰의 감정을 성공적으로 분류함
 - 다양한 시각화를 통해 항공사별 연도별 평가 변화를 파악 가능
 - 종합 지표를 활용한 평가로 고객에게 신뢰도 높은 항공사 추천 가능

---

## 6. 개발 환경

| 항목           | 버전     |
|--------------|--------|
| Python       | 3.12   | 
| Torch        | 2.3.0  |
| Transformers | 4.44.2 |
| pandas       | 2.2.3  |
| numpy        | 1.26.4 
| scikit-learn | 1.6.1  


---

## 7. 참고 자료

- https://www.kaggle.com/datasets/juhibhojani/airline-reviews
- https://www.kaggle.com/

---

  