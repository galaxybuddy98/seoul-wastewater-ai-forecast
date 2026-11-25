# 서울시 물재생센터 유입수질 예측

과거 30일의 수질·기온·강우 자료로 서울시 4개 물재생센터의 **다음 날 BOD·T-N·T-P를 예측**한 시계열 머신러닝 프로젝트입니다. 대규모 강우 자료의 품질을 검증하고 LightGBM·XGBoost·LSTM을 공정하게 비교해 목표별 XGBoost 모델을 선정했습니다.

| 핵심 성과 | 결과 |
|---|---|
| 최종 모델 | 목표별 30일 입력 기반 통합 XGBoost |
| Validation 개선 | Persistence 기준모델 대비 평균절대오차(MAE) BOD **20.02%**, T-N **22.05%**, T-P **21.87%** 개선 |
| 데이터 처리 | 서울시 10분 강우 **38,322,021행**, 유입수질 **28,828행** |
| 신뢰성 검증 | 강우 650 관측소-일·모델 입력 19,584개 값 독립 복원, 자동 테스트 107개 통과 |

## 1. 프로젝트 소개

- **예측 목표:** 처리장·유입수 관측 계통별 다음 날(`t+1`) BOD·T-N·T-P
- **평가 구조:** 원본 11개 계통 → 학습 가능 10개 → Validation/Test 평가 가능 7개
- **시간 분할:** Train 2016년부터 2022년까지, Validation 2023년부터 2024년까지, Test 2025-01-01부터 2026-09-30까지
- **선택 원칙:** Validation MAE로 모델을 선택하고 Test는 선택 후 참고 성능으로만 사용

| 목표 | 최종 입력 구성 | Validation MAE | Persistence 대비 개선 | Test MAE¹ | Test R²¹ |
|---|---|---:|---:|---:|---:|
| BOD | 수질·기온·일강우량·계절·요일 | **23.2784** | **20.02%** | 22.4176 | 0.6028 |
| T-N | 수질·기온·일강우량·강우 파생변수·요일 | **3.1977** | **22.05%** | 3.3942 | 0.5981 |
| T-P | 수질·기온·일강우량·요일 | **0.5093** | **21.87%** | 0.5294 | 0.4027 |

¹ 전체 7개 평가 계통을 합친 통합 지표입니다. [최종 성능표](reports/final/model_metrics.csv)에서 재확인할 수 있습니다.

## 2. 대표 Test 예측 결과

난지물재생센터의 대표 예측 사례로, 전체 Test 기간의 실제 관측값과 예측값을 비교했습니다. 성능이 양호한 개별 계통의 결과이며, 평활화 및 보간을 적용하지 않았습니다.

### T-N — 난지물재생센터 처리장 구분없음

Test MAE **2.6237mg/L**, R² **0.7432**, n=638, Persistence 대비 MAE **23.68% 개선**.

![T-N 난지 Test 전체 기간 예측](docs/images/tn_test_nanj_full.png)

<details>
<summary><strong>BOD·T-P 대표 그래프 보기</strong></summary>

### BOD — 난지물재생센터 처리장 구분없음

Test MAE **20.8214mg/L**, R² **0.5966**, n=638, Persistence 대비 MAE **22.42% 개선**.

![BOD 난지 Test 전체 기간 예측](docs/images/bod_test_nanj_full.png)

### T-P — 난지물재생센터 처리장 구분없음

Test MAE **0.2439mg/L**, R² **0.6535**, n=638, Persistence 대비 MAE **29.63% 개선**.

![T-P 난지 Test 전체 기간 예측](docs/images/tp_test_nanj_full.png)

</details>

## 3. 데이터 파이프라인

![서울시 물재생센터 유입수질 예측 파이프라인](docs/images/data_pipeline.svg)


| 단계 | 핵심 처리 |
|---|---|
| 수집 | 서울시 수질·처리량 CSV, 기상청 ASOS API, 서울시 10분 강우 자료 |
| 정제·결합 | 유입수 필터링, 일강우 집계, 센터·처리장·관측 계통별 결합 |
| 입력 생성 | 계통별 과거 30일 입력과 다음 날 목표 구성 |
| 검증 | 결측·중복·원본 대조, 미래 정보 누수와 계통 혼합 차단 |
| 학습·평가 | 시간순 분할 후 Persistence·LightGBM·XGBoost·LSTM 비교 |

| 자료 | 기간·규모 |
|---|---|
| 유입수질 | 2016-01-01부터 2026-09-30까지, 수질 원본 74,917행 중 유입수 28,828행 |
| 처리량 | 원본 48,771행, 정확 키 결합 18,443행 |
| 기온 | 기상청 ASOS 서울 108 중심, 수질 28,828행에 결합 |
| 강우 | 서울시 10분 관측 파일 944개, 38,322,021행, 관측소 69개 |

원본 농도는 임의 평균·보간하지 않았고, 미확인 강우 결측은 `NaN`으로 유지했습니다. 처리량 `1차+2차`는 공식 유입유량이 아닌 검증 가설입니다. 자세한 기준은 [수질 품질 보고서](reports/data_quality/water_quality.md)와 [센터·계통 통합 규칙](docs/data/integration_rules.md)을 참고하세요.

## 4. 데이터 신뢰성 검증

기존 전처리 함수를 호출하지 않는 별도 감사 경로로 원본과 최종 데이터셋을 대조했습니다.

| 감사 항목 | 결과 |
|---|---:|
| 원본 파일 SHA-256 기준선 | CSV 947개, 약 1.24GB |
| 수질 원본-가공 전수 대조 | 28,828/28,828행 일치 |
| 강우 독립 재계산 | 650/650 관측소-일 일치 |
| 30일 모델 입력 독립 복원 | 19,584/19,584개 값 일치 |
| 계통 혼합·미래 날짜·분할 오류 | 0건 |
| 자동 테스트 | 107개 통과 |

강우 데이터는 고정 시드 기반 층화 표본 추출을 통해 독립 재계산했으며, 원본 파일의 무결성 관리를 위해 SHA-256 기준선을 구축했습니다. 상세 검증 절차는 [독립 전처리 감사](reports/data_quality/independent_preprocessing_audit.md)를 참고하세요.

## 5. 모델 비교 및 개선 실험

과거 수질·ASOS 기온 자료를 활용하여 동일한 30일 입력과 공통 Validation 표본에서 Persistence·LightGBM·XGBoost·LSTM의 평균절대오차(MAE)를 비교했습니다.

| 목표 | Persistence | LightGBM | XGBoost | LSTM |
|---|---:|---:|---:|---:|
| BOD | 29.1064 | 24.0091 | **23.7582** | 24.6479 |
| T-N | 4.1024 | 3.3801 | **3.3528** | 3.3975 |
| T-P | 0.6519 | **0.5294** | 0.5303 | 0.5368 |

- **서울시 강우:** XGBoost Validation MAE를 BOD 1.24%, T-N 4.40%, T-P 3.31% 개선했습니다.
- **계절·요일:** BOD는 계절+요일, T-N·T-P는 요일 추가 시 소폭 개선됐습니다.
- **부유물질(SS):** XGBoost T-N만 0.31% 개선되어 최종 공통 피처로 채택하지 않았습니다.
- **고농도 가중:** 전체 성능 악화를 제한했지만 고농도 MAE 10% 개선 목표에는 미달했습니다.

상세 결과는 [모델 비교](reports/experiments/comparison/model_comparison.md), [달력 피처](reports/experiments/xgboost/calendar_features.md), [SS 피처](reports/experiments/shared_features/ss_feature.md), [최종 예측 분석](reports/final/prediction_analysis.md)에서 확인할 수 있습니다.

## 6. 한계 및 개선 방향

| 한계 | 개선 방향 |
|---|---|
| 고농도 BOD·T-P 과소예측 | 비대칭 손실함수 및 예측구간 검토 |
| 계통별 성능 차이 | 계통별 독립 모델과 통합 모델 비교 |
| 강우 관측소 대표성 | 하수처리구역 기반 GIS 검증 |
| 운영 환경 적용 | 데이터 공개 시점·처리량 정의·모델 모니터링 검증 |
| 반복 확인된 Test 구간 | 미사용 기간 또는 rolling-origin 추가 검증 |

고농도 BOD 구간에서 과소예측 경향을 확인했으며, 계통별 오차 특성을 분석했습니다. 상세 결과는 [오차 분석](reports/final/error_analysis.md)을 참고하세요.

## 7. 실행 방법

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

API 수집에는 `SEOUL_API_KEY`, `DATA_GO_KR_API_KEY` 환경변수가 필요합니다. 실제 키를 담은 `.env`와 키 예시 파일은 공개 저장소에서 제외합니다. 원본 CSV·Parquet·NPZ·모델 가중치도 Git에서 제외되므로 전체 재현에는 서울시·기상청 외부 원자료를 별도로 준비해야 합니다.

### 주요 실행 명령어

```powershell
python -m scripts.data_collection.collect_asos_daily
python -m scripts.preprocessing.process_seoul_rainfall
python -m scripts.preprocessing.build_training_dataset
python -m scripts.training.xgboost.train_xgboost_comparison --train-one --config C --window 30 --target bod_mg_l
python -m scripts.validation.run_preprocessing_audit
```

## 8. 프로젝트 구조

```text
├─ configs/                 # 관측소·수집·통합 설정
├─ docs/                    # data·modeling·project·validation 문서
├─ reports/
│  ├─ final/                # 최종 성능·예측·오차 분석
│  ├─ data_quality/         # 데이터 품질·독립 감사
│  ├─ experiments/          # comparison·모델·공통 피처 실험
│  └─ archive/              # 과거 검토 기록
├─ scripts/
│  ├─ data_collection/      # API·기상 수집
│  ├─ preprocessing/        # 강우·학습 데이터셋 생성
│  ├─ training/             # lightgbm·xgboost·lstm·shared
│  ├─ evaluation/           # 모델 비교·오차 분석
│  ├─ visualization/        # 최종 예측 시각화
│  └─ validation/           # 데이터·산출물 검증
├─ src/
│  ├─ audit/                # 독립 감사 로직
│  ├─ data/                 # API·캐시·서울시 강우 처리
│  ├─ evaluation/           # 평가 키·시각화 검증
│  ├─ models/               # lightgbm·xgboost·lstm·shared
│  └─ preprocessing/        # 스키마·결합·윈도 생성
└─ tests/                   # 기능·알고리즘별 회귀 테스트
```
