# 최종 수질 예측 간소화 시각화

## 구성

기존 모델과 분석 결과는 변경하지 않고, 시계열 그래프에서 다음 두 선만 표시했다.

- 실제 관측값: 파란색 실선
- 최종 XGBoost 예측값: 주황색 실선

Persistence·LightGBM·LSTM은 새 시계열에서 제외했다. 기존 알고리즘별 MAE·RMSE·R² 비교표와 기존 이미지는 그대로 보존했다.

## 생성 결과

- Validation/Test × BOD·T-N·T-P × 7개 계통 개별 그림: 42개
- 7개 계통 전체 패널: 6개
- Validation/Test 90일 확대 패널: 6개
- BOD 고농도 전후 7일 패널: 2개
- 합계: 300dpi PNG 56개

모든 그림은 계통별 독립 시계열로 만들었다. 날짜 공백은 선을 끊었으며 결측값 보간을 수행하지 않았다. MAE·R²·표본 수는 별도 상자 대신 제목 아래 한 줄로 배치했다.

## 대표 그래프 8개와 README 권장 순서

1. `zoom_bod_mg_l_validation.png` — Validation 실제 변화량으로 선정한 90일 BOD, 7계통 비교
2. `high_bod_context_validation.png` — BOD 180mg/L 이상 급등 전후 7일과 과소예측
3. `series_bod_mg_l_validation_series_2b498f0fc9d3.png` — 중랑 제2처리장 BOD 취약 사례
4. `zoom_tn_mg_l_validation.png` — T-N 90일 변화 추종
5. `series_tn_mg_l_validation_series_38b56308fe33.png` — 탄천 처리장 구분없음 T-N 취약 사례
6. `zoom_tp_mg_l_validation.png` — T-P 90일 변화 및 급등 축소
7. `series_tp_mg_l_validation_series_2b498f0fc9d3.png` — 중랑 제2처리장 T-P 취약 사례
8. `overview_bod_mg_l_test.png` — 반복 확인된 참고 Test의 7계통 BOD

README에서는 1~2로 핵심 성능과 한계를 먼저 보여주고, 3~7로 항목·계통별 차이를 설명한 뒤, 8에서 참고 Test 결과와 한계를 명시하는 순서를 권장한다. 기존 다중 모델 시계열은 이 목록에서 제외했다.

## 기간 선정과 주의사항

확대 구간은 이전 시각화와 동일하게 모델 오차나 Test 결과가 아닌 Validation 실제값의 90일 총 절대 변화량을 기준으로 선정했다.

- BOD: 2024-04-08~2024-07-06
- T-N: 2023-04-14~2023-07-12
- T-P: 2024-03-25~2024-06-22

Test 확대 그림은 같은 월·일의 2025년 구간을 사용한다. BOD 고농도는 실제값을 이용한 사후 분류이며 예측 시점에 알려진 정보로 해석하지 않는다.

## 저장 및 재현

- 이미지: `results/figures/final_model_simplified_2026-11-14/`
- 그림 목록: `reports/final_model_simplified_manifest_2026-11-14.csv`
- 실행 요약: `reports/final_model_simplified_summary_2026-11-14.json`
- 재현 스크립트: `scripts/visualization/visualize_simplified_final_predictions.py`
- 실행 명령: `python -m scripts.visualization.visualize_simplified_final_predictions`

모델 재학습, 원본 CSV 재처리, API 호출, 기존 이미지 삭제, README 수정 및 GitHub Push는 수행하지 않았다.
