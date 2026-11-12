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


## 학습 데이터셋 변수

| 변수 | 의미 |
|---|---|
| center_id | 센터명에서 결정적으로 생성한 비식별 ID |
| series_id | 센터·원본 처리장·유입 계통 조합의 결정적 ID |
| source_plant | 원본 처리장 구분 보존값 |
| observation_scope | 원본 유입 관측 계통이며 대표성 미확인임을 표시 |
| mean_temperature_c | ASOS 일평균기온 |
| precipitation_mm | ASOS 일강수량, 공란은 결측 보존 |
| precipitation_lag_1d~3d | 같은 센터의 1~3일 전 강수 |
| precipitation_sum_3d/7d | t일까지 누적강수 검토 열, 기본 모델 입력에서는 제외 |
| flow_link_status | 날짜·센터·처리장 정확키 연결 또는 대응 불명확 |
| weather_link_status | 센터·날짜 기상 연결 상태 |
| target_date | 예측 대상일이며 시간 분할 기준 |
| input_start_date/input_end_date | 연속 입력창 경계 |
# 강수 전처리 실험 변수

| 변수 | 형식 | 의미 |
|---|---|---|
| `precipitation_raw_mm` | float | ASOS `sumRn` 수치 원본. 공란은 NaN |
| `precipitation_raw_blank` | bool | 원본 `sumRn` 공란 여부 |
| `precipitation_processed_mm` | float | 검증 규칙 적용 후 일강수량. 미확인 공란은 NaN |
| `precipitation_missing` | int8 | 처리 강수량 NaN 표시(1=결측) |
| `precipitation_processing_reason` | string | 수치 보존 또는 공란 미확인 사유 |
| `precipitation_processing_evidence` | string | 변환·보류 근거 |
| `precipitation_lag_1d`~`3d` | float | 해당 일 기준 1~3일 전 처리 강수량 |
| `precipitation_sum_3d`, `precipitation_sum_7d` | float | 해당 일을 포함한 최근 3·7일 합계. 기간 내 NaN이 있으면 NaN |

`sumRn` 공란은 공식 의미가 확정되지 않아 0mm로 일괄 변환하지 않는다. 이 변수들은 강수 실험용이며 기존 기준 데이터셋을 대체하지 않는다.

## ASOS 시간자료 검증 변수

| 변수 | 형식 | 의미 |
|---|---|---|
| `timestamp` | datetime | ASOS 시간자료 관측 시각. `rn`은 해당 시각 직전 구간으로 판정 |
| `hourly_precipitation_mm` | float | API `rn`, 단위 mm. 공란은 NaN 유지 |
| `rainfall_qc_flag` | string | 실제 응답 `rnQcflg`. 0·1·9·공란 원문 보존 |
| `hourly_precipitation_sum_mm` | float | 00시를 전날에 귀속한 `00:01~24:00` 시간합계 |
| `validated_precipitation_mm` | float | 공식 일자료 우선의 최종 검증 강수량 |
| `rainfall_status` | string | A 일자료/시간자료 확인, B 무강수 확인, C 미량·미확인, D 오류·누락 |
| `hourly_row_count` | int | 일계 구간에 반환된 시간 시각 행 수 |
| `hourly_numeric_count` | int | 수치 `rn` 건수. 겨울 3시간 제공 특성 때문에 24를 요구하지 않음 |
| `hourly_qc_missing_count` | int | QC 9 건수 |
| `hourly_duplicate_timestamp_count` | int | 일계 구간 중복 시각 건수 |
| `rainfall_decision_reason` | string | 최종값 확정·보류 근거 |
# 서울시 강우량 추가 변수

서울시 강우량 관측망 변수는 기존 ASOS 변수와 구분하기 위해 `seoul_rainfall_` 접두사를 사용한다.

| 변수 | 의미 | 단위/상태 |
|---|---|---|
| `seoul_rainfall_station_id` | 관측소명에서 생성한 안정 식별자 | 문자열 |
| `seoul_rainfall_mm` | 144개 10분 구간이 완전한 날의 합 | mm/day |
| `seoul_rainfall_max_1h_mm` | 같은 완전일의 연속 6개 구간 최대 합 | mm/hour |
| `seoul_rain_day` | 확정 일강우량이 0보다 큰지 여부 | 0/1/NaN |
| `seoul_rainfall_lag_1d_mm`~`lag_3d_mm` | 관측소별 과거 1~3일 강우 | mm |
| `seoul_rainfall_sum_3d_mm`, `sum_7d_mm` | 모든 구성일이 확정된 경우의 누적 강우 | mm |
| `seoul_rainfall_missing` | 당일 일강우 미확정 여부 | 0/1 |
| `seoul_rainfall_quality_status` | 완전, 기록 없음, 구간 불완전, 중복 충돌 | 범주 |

센터-관측소 대응은 현재 `provisional`이며 ASOS 108 공통 기상자료를 대체하지 않는다. 상세 판정은 `reports/seoul_rainfall_report.md`를 따른다.
