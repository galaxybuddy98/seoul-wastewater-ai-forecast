# XGBoost·LSTM 및 LightGBM 3종 모델 비교

## 평가 계약

- 목표: 처리장·유입 관측 계통별 다음 날 BOD·T-N·T-P.
- 공통 입력: 과거 수질 3종+ASOS 기온(B), 7·14·30일.
- 분할: Train 2016~2022, Validation 2023~2024, Test 2025~2026-09-30 (`target_date` 기준).
- 원본 11개 관측 계통 중 유효 연속 입력창은 Train 10개, Validation/Test 7개 계통에서 생성됐다.
- 모델 선택은 Validation MAE만 사용했다. Test는 반복 확인된 참고 구간으로 완전히 독립적인 최종 홀드아웃이 아니다.
- XGBoost 3.4.1과 CPU PyTorch 2.15.0을 사용했으며 CUDA 장치는 없었다.

서울시 강우 실험의 입력 구성은 다음과 같다.

| 식별자 | 입력 구성 |
|---|---|
| P | Persistence 기준모델 |
| A | 과거 수질(BOD·T-N·T-P) |
| B | 수질·ASOS 기온 |
| C | B + 서울시 일강우량 |
| D | C + 1~3일 전 강우량 + 3·7일 누적강우량 |
| E | D + 최대 1시간 강우량 |

이 C·D는 ASOS `sumRn` 결측 허용 실험의 C·D와 다른 구성이다.

## Validation 선택 결과

선택 윈도: `{"bod_mg_l": {"lightgbm": 14, "lstm": 30, "xgboost": 30}, "tn_mg_l": {"lightgbm": 30, "lstm": 30, "xgboost": 30}, "tp_mg_l": {"lightgbm": 30, "lstm": 30, "xgboost": 30}}`

| target   | split      | algorithm   |   eval_rows |       mae |     rmse |        r2 |   persistence_mae |   persistence_mae_improvement_pct |
|:---------|:-----------|:------------|------------:|----------:|---------:|----------:|------------------:|----------------------------------:|
| bod_mg_l | validation | lightgbm    |        5056 | 23.9571   | 40.4849  | 0.411225  |         29.1064   |                           17.6913 |
| bod_mg_l | validation | lstm        |        5056 | 24.6479   | 40.963   | 0.397237  |         29.1064   |                           15.318  |
| bod_mg_l | validation | persistence |        5056 | 29.1064   | 49.0325  | 0.136365  |         29.1064   |                            0      |
| bod_mg_l | validation | xgboost     |        5056 | 23.7582   | 40.0404  | 0.424083  |         29.1064   |                           18.3747 |
| tn_mg_l  | validation | lightgbm    |        5056 |  3.38007  |  5.25621 | 0.53853   |          4.10237  |                           17.6068 |
| tn_mg_l  | validation | lstm        |        5056 |  3.3975   |  5.29652 | 0.531424  |          4.10237  |                           17.182  |
| tn_mg_l  | validation | persistence |        5056 |  4.10237  |  6.29854 | 0.337359  |          4.10237  |                            0      |
| tn_mg_l  | validation | xgboost     |        5056 |  3.35275  |  5.22731 | 0.54359   |          4.10237  |                           18.2728 |
| tp_mg_l  | validation | lightgbm    |        5056 |  0.529427 |  1.21362 | 0.295628  |          0.651866 |                           18.7829 |
| tp_mg_l  | validation | lstm        |        5056 |  0.536784 |  1.20334 | 0.307513  |          0.651866 |                           17.6543 |
| tp_mg_l  | validation | persistence |        5056 |  0.651866 |  1.38483 | 0.0828754 |          0.651866 |                            0      |
| tp_mg_l  | validation | xgboost     |        5056 |  0.530274 |  1.21836 | 0.290113  |          0.651866 |                           18.6529 |

## 동일 평가 표본 Test 참고 성능

| target   | split   | algorithm   |   eval_rows |       mae |     rmse |       r2 |   persistence_mae |   persistence_mae_improvement_pct |
|:---------|:--------|:------------|------------:|----------:|---------:|---------:|------------------:|----------------------------------:|
| bod_mg_l | test    | lightgbm    |        3831 | 23.2484   | 35.5794  | 0.581678 |          26.7611  |                           13.1261 |
| bod_mg_l | test    | lstm        |        3831 | 24.0063   | 36.9049  | 0.54993  |          26.7611  |                           10.2941 |
| bod_mg_l | test    | persistence |        3831 | 26.7611   | 40.1466  | 0.467388 |          26.7611  |                            0      |
| bod_mg_l | test    | xgboost     |        3831 | 22.8668   | 34.9477  | 0.5964   |          26.7611  |                           14.5522 |
| tn_mg_l  | test    | lightgbm    |        3831 |  3.61564  |  5.30253 | 0.542845 |           4.53337 |                           20.2439 |
| tn_mg_l  | test    | lstm        |        3831 |  3.77486  |  5.50437 | 0.507379 |           4.53337 |                           16.7317 |
| tn_mg_l  | test    | persistence |        3831 |  4.53337  |  6.44104 | 0.325459 |           4.53337 |                            0      |
| tn_mg_l  | test    | xgboost     |        3831 |  3.57337  |  5.2654  | 0.549225 |           4.53337 |                           21.1764 |
| tp_mg_l  | test    | lightgbm    |        3831 |  0.555427 |  1.01537 | 0.372802 |           0.67268 |                           17.4307 |
| tp_mg_l  | test    | lstm        |        3831 |  0.580882 |  1.0315  | 0.352715 |           0.67268 |                           13.6466 |
| tp_mg_l  | test    | persistence |        3831 |  0.67268  |  1.06105 | 0.315099 |           0.67268 |                            0      |
| tp_mg_l  | test    | xgboost     |        3831 |  0.555747 |  1.01967 | 0.367476 |           0.67268 |                           17.3831 |

전체 7·14·30일 윈도별 공통 성능은 `reports/model_algorithm_common_metrics.csv`에 보존했다.

## 서울시 강우량 추가 효과

| target   | split      | algorithm     |   eval_rows |       mae |     rmse |        r2 |   persistence_mae |   persistence_mae_improvement_pct | base_algorithm   | input_config   |
|:---------|:-----------|:--------------|------------:|----------:|---------:|----------:|------------------:|----------------------------------:|:-----------------|:---------------|
| bod_mg_l | validation | lightgbm_B    |        5056 | 24.0091   | 40.5135  | 0.410392  |         29.1064   |                           17.5128 | lightgbm         | B              |
| bod_mg_l | validation | lightgbm_rain |        5056 | 23.6926   | 40.0748  | 0.423094  |         29.1064   |                           18.6    | lightgbm         | C              |
| bod_mg_l | validation | lstm_B        |        5056 | 24.6479   | 40.963   | 0.397237  |         29.1064   |                           15.318  | lstm             | B              |
| bod_mg_l | validation | lstm_rain     |        5056 | 24.1891   | 39.8909  | 0.428375  |         29.1064   |                           16.8944 | lstm             | C              |
| bod_mg_l | validation | persistence   |        5056 | 29.1064   | 49.0325  | 0.136365  |         29.1064   |                            0      | persistence      | P              |
| bod_mg_l | validation | xgboost_B     |        5056 | 23.7582   | 40.0404  | 0.424083  |         29.1064   |                           18.3747 | xgboost          | B              |
| bod_mg_l | validation | xgboost_rain  |        5056 | 23.4625   | 39.6539  | 0.435148  |         29.1064   |                           19.3905 | xgboost          | C              |
| tn_mg_l  | validation | lightgbm_B    |        5056 |  3.38007  |  5.25621 | 0.53853   |          4.10237  |                           17.6068 | lightgbm         | B              |
| tn_mg_l  | validation | lightgbm_rain |        5056 |  3.24471  |  5.04335 | 0.575149  |          4.10237  |                           20.9065 | lightgbm         | E              |
| tn_mg_l  | validation | lstm_B        |        5056 |  3.3975   |  5.29652 | 0.531424  |          4.10237  |                           17.182  | lstm             | B              |
| tn_mg_l  | validation | lstm_rain     |        5056 |  3.25858  |  5.04456 | 0.574944  |          4.10237  |                           20.5683 | lstm             | E              |
| tn_mg_l  | validation | persistence   |        5056 |  4.10237  |  6.29854 | 0.337359  |          4.10237  |                            0      | persistence      | P              |
| tn_mg_l  | validation | xgboost_B     |        5056 |  3.35275  |  5.22731 | 0.54359   |          4.10237  |                           18.2728 | xgboost          | B              |
| tn_mg_l  | validation | xgboost_rain  |        5056 |  3.2051   |  4.98758 | 0.584493  |          4.10237  |                           21.8719 | xgboost          | E              |
| tp_mg_l  | validation | lightgbm_B    |        5056 |  0.529427 |  1.21362 | 0.295628  |          0.651866 |                           18.7829 | lightgbm         | B              |
| tp_mg_l  | validation | lightgbm_rain |        5056 |  0.516136 |  1.20562 | 0.304881  |          0.651866 |                           20.8218 | lightgbm         | C              |
| tp_mg_l  | validation | lstm_B        |        5056 |  0.536784 |  1.20334 | 0.307513  |          0.651866 |                           17.6543 | lstm             | B              |
| tp_mg_l  | validation | lstm_rain     |        5056 |  0.521864 |  1.194   | 0.318216  |          0.651866 |                           19.9431 | lstm             | C              |
| tp_mg_l  | validation | persistence   |        5056 |  0.651866 |  1.38483 | 0.0828754 |          0.651866 |                            0      | persistence      | P              |
| tp_mg_l  | validation | xgboost_B     |        5056 |  0.530274 |  1.21836 | 0.290113  |          0.651866 |                           18.6529 | xgboost          | B              |
| tp_mg_l  | validation | xgboost_rain  |        5056 |  0.51272  |  1.20068 | 0.310574  |          0.651866 |                           21.3458 | xgboost          | C              |

강우 비교는 BOD C/30일, T-N E/30일, T-P C/30일을 사용했다. XGBoost와 LightGBM은 NaN native 처리를 사용했고 LSTM은 Train 중앙값 대치와 변수별 결측 표시를 함께 사용했다. ASOS 강수량은 제외했다.

## BOD 고농도 진단

| algorithm   |   threshold |   rows |     mae |    rmse |        r2 |   underprediction_rate |   mean_bias |
|:------------|------------:|-------:|--------:|--------:|----------:|-----------------------:|------------:|
| lightgbm    |         180 |    362 | 68.3571 | 87.6834 | -0.92263  |               0.969613 |    -67.7756 |
| lstm        |         180 |    362 | 62.0985 | 83.0899 | -0.726463 |               0.922652 |    -60.6549 |
| persistence |         180 |    362 | 54.8887 | 75.0463 | -0.408376 |               0.723757 |    -31.2224 |
| xgboost     |         180 |    362 | 66.4413 | 86.3984 | -0.86669  |               0.966851 |    -65.7681 |

고농도 기준은 기존과 동일한 Train BOD 90백분위다. 이 단계에서는 원인 진단만 수행했으며 가중 손실·표본 가중은 적용하지 않았다.

## 학습시간

| algorithm   | input_config        |   model_count |   total_training_seconds |   mean_training_seconds |   max_training_seconds |
|:------------|:--------------------|--------------:|-------------------------:|------------------------:|-----------------------:|
| lstm        | B                   |             9 |                 149.715  |                16.635   |                29.7036 |
| lstm        | C                   |             2 |                  47.7127 |                23.8563  |                25.3553 |
| lstm        | E                   |             1 |                  23.2142 |                23.2142  |                23.2142 |
| xgboost     | B                   |             9 |                  46.6406 |                 5.18228 |                11.8131 |
| xgboost     | C                   |             2 |                  41.409  |                20.7045  |                21.0774 |
| xgboost     | E                   |             1 |                  36.5159 |                36.5159  |                36.5159 |
| lightgbm    | historical_full_run |           nan |                  55.61   |               nan       |               nan      |

LightGBM 55.61초는 기존 전체 기준모델 실행 기록으로 B 구성만의 순수 합계가 아니므로 XGBoost·LSTM 합계와 직접 등가 비교할 수 없다.

## 해석과 제한

- 강우량 추가 효과는 예측 연관성이지 인과효과가 아니다.
- 센터별 서울시 강우 관측소 매핑은 `provisional`이다.
- LSTM 단일 seed 결과이므로 신경망 seed 안정성은 추가 검증이 필요하다.
- 실제 운영에서 입력일 t 강우·기온의 공개 지연을 별도로 확인해야 한다.

## 검증 상태

- 리뷰 전 전체 회귀 테스트는 76개가 통과했다.
- 독립 리뷰에서 Critical 0건, Important 2건(동일 조건 평가키 strict 검증, `tabulate` 의존성)이 확인되어 모두 수정했다.
- 리뷰 수정 후 영향 범위 테스트 8개, `compileall`, `pip check`, `git diff --check`는 통과했다.
- 리뷰 수정 후 전체 회귀 재실행은 Windows 가상 메모리 고갈로 pytest 수집 중 `MemoryError`가 발생했다. 코드 실패로 판정하지 않았으며, 메모리 여유 확보 후 `python -m pytest -q` 재확인이 필요하다.

후속 최종 검증에서 전체 자동 테스트 107개가 통과했다. 위 `MemoryError`는 이 보고서 작성 당시의 검증 이력으로 보존한다.
