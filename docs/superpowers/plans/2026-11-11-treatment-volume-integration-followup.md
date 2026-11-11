# 처리량 합산·유입수질 통합 후속 검증 계획

## 목표

기존 CSV 분석과 API 수집 구조를 유지하면서 1차+2차 처리량 합산 가설, 센터별 대표 유입수질 구성 가능성, 결측·이상치가 7·14·30일 입력창에 미치는 영향을 검증한다. 모델 학습은 수행하지 않는다.

## 전역 제약

- 원본 CSV와 API 수집·캐시 코드는 수정하지 않는다.
- 목표는 다음 날 유입 BOD·T-N·T-P이며 방류수는 제외한다.
- 처리량 합산과 수질 농도 통합은 별개 규칙이다.
- 처리량 합계는 공식 비중복성이 확인되기 전까지 가설 변수이며 실제 유입유량으로 부르지 않는다.
- 잠긴 테스트 구간은 규칙 선택·이상치 경계 산정에 사용하지 않는다.

## 검토 초점

- 한 성분 결측 시 합계가 0으로 대체되는 오류
- 보조시설과 처리장의 중복 합산
- 처리장 구분없음과 개별 처리장의 중복 농도 통합
- 입력창과 +1일 목표 사이의 날짜 단절
- 통계적 이상치를 실제 오류로 자동 삭제하는 처리

### Task 1: 공식 정의와 센터 구조 조사

- 서울시 공식 자료에서 처리량 필드·시설 범위·수질 API 구분값을 확인한다.
- 공식 근거, 데이터상 관찰, 추론, 미확인을 분리한다.
- 센터별 직접 사용·유량 통합·처리장별 운영 중 하나를 조건부 판정한다.

### Task 2: 처리량 합산 가설 구현

**Files:**
- Create: src/preprocessing/flow.py
- Create: tests/test_flow.py
- Modify: scripts/analyze_external_csv.py

**Interfaces:**
- Produces: add_treatment_total_hypothesis(frame) -> DataFrame
- Produces: summarize_treatment_hypothesis(frame) -> DataFrame

- 두 성분이 있을 때만 합계를 만들고 원 성분을 보존하는 실패 테스트를 작성한다.
- 한 성분 결측 시 합계를 결측으로 유지하는 실패 테스트를 작성한다.
- 테스트 실패 원인을 확인한 뒤 최소 구현한다.
- 처리장별 합산 가설 통계를 분석 산출물로 기록한다.

### Task 3: 연속 입력창 표본 검증

**Files:**
- Modify: src/preprocessing/profile.py
- Modify: tests/test_profile.py
- Modify: scripts/analyze_external_csv.py

**Interfaces:**
- Produces: count_next_day_window_samples(...) -> DataFrame

- 과거 L일과 +1일 목표가 모두 연속·관측된 경우만 세는 실패 테스트를 작성한다.
- 목표일 기준 Train/Validation/Test 표본 수가 분리되는 실패 테스트를 작성한다.
- 직접 대표 후보와 처리장별 시계열을 분리해 7·14·30일 표본 수를 산출한다.
- 테스트 구간 표본 수는 기술 통계로만 보고 규칙 선택에 사용하지 않는다.

### Task 4: 품질 후속 산출물

- 이상치 후보를 센터·연도·처리장·변수별로 집계한다.
- BOD와 처리량 0값을 원자료 식별키와 함께 보존한다.
- 처리장별 관측 시작·중단 기간과 입력창 영향을 교차검증한다.

### Task 5: 문서·정책 갱신

- AGENTS.md, 통합 규칙, 데이터 사전, 요구사항, 프로젝트 계획, 품질 보고서를 갱신한다.
- 합산 가설·실제 유입유량·센터 농도 통합의 의미를 분리한다.
- 센터별 확정·조건부·보류 규칙과 대안을 기록한다.
- ASOS 후속 분석은 실제 자료가 준비된 뒤 수행하도록 설계만 유지한다.

### Task 6: 검증

- 신규 테스트의 RED→GREEN을 기록한다.
- python -m scripts.analyze_external_csv를 실행한다.
- 전체 pytest -q를 실행한다.
- 원본 SHA-256, 행 수, 기간이 기존 값과 일치하는지 확인한다.
- 문서 수치와 생성 CSV를 교차검증한다.
