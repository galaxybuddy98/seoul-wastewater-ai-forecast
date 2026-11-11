# 실제 CSV 데이터 사전

## 수질현황

CP949, 74,917행×11열, 2016-01-01~2026-09-30이다.

| 원본 열 | 자료형 | 표준 열 | 확인 |
|---|---|---|---|
| 측정일자 | object | date | 변환 실패 0 |
| 물재생센터명칭 | object | facility | 4개 센터 |
| 처리장구분 | object | plant | 제1~4, 구분없음, 시설현대화 |
| 계통구분 | object | stream_type | 유입수·방류 |
| BOD | float64 | bod_mg_l | 유입 목표, 중랑 0값 19 |
| TOC(2021년 1월 이전자료는 COD) | float64 | toc_or_cod_value | 날짜별 COD/TOC 라벨 유지 |
| SS | float64 | ss_mg_l | 보조 수질 |
| T-N | float64 | tn_mg_l | 유입 목표 |
| TP | float64 | tp_mg_l | 의미상 T-P, 원본명 유지 |
| PH | float64 | ph | 장기 전체 결측 존재 |
| 총대장군균 | float64 | total_coliform | 결측·0 존재 |

## 하수처리량

CP949, 48,771행×5열, 같은 기간이다.

| 원본 열 | 자료형 | 표준 열 | 확인 |
|---|---|---|---|
| 측정일자 | object | date | 변환 실패 0 |
| 물재생센터명칭 | object | facility | 4개 센터 |
| 처리장구분 | object | plant | 처리장 외 중계펌프장·오니처리장 포함 |
| 1차하수처리량 | float64 | primary_treatment_volume | 0·결측 다수, 유입유량 단정 금지 |
| 2차하수처리량 | float64 | secondary_treatment_volume | 1차와 합산 금지, 0값 93 |

단위와 계측 경계는 API 메타데이터 또는 담당기관 확인이 필요하다.


## 후속 계산 변수

| 변수 | 형식 | 의미 | 사용 제한 |
|---|---|---|---|
| total_treatment_volume_hypothesis | float64 | 같은 처리장·일자의 1차+2차 처리량 계산 가설 | 두 성분 모두 관측 시만 계산, 공식 총량·실제 유입유량 아님 |
| series_basis | string | 센터 대표 후보 또는 처리장별 대안 | 확정 통합 규칙이 아니라 표본 가용성 분석 구분 |
| window_days | int | 7·14·30일 연속 과거 관측기간 | +1일 목표까지 날짜 연속·비결측일 때만 표본 인정 |
| source_tp_column | string | 원본 TP 표시 보존 | 분석 변수 tp_mg_l, 문서 표기 T-P와 매핑 |

API의 원 필드는 수질 계통 STATS_SE, 목표값 BOD, TN, TP이며 샘플 응답에서 유입수와 방류가 구분된다. TP는 T-P 분석 변수로 매핑하되 원 필드명을 보존한다.
