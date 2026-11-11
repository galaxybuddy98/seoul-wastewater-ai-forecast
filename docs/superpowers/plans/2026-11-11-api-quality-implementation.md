# API 수집 및 데이터 품질 검증 구현 계획

> **에이전트 작업자 필수 절차:** 작업별로 `superpowers:subagent-driven-development` 또는 `superpowers:executing-plans`를 사용하고, 체크박스로 진행 상태를 추적한다.

**목표:** 인증키와 CSV가 없어도 자동 테스트할 수 있고, 제공되면 API 수집·CSV 대조·품질 보고를 재현할 수 있는 데이터 계층을 만든다.

**구조:** API 전송, 서비스별 응답 해석, 원본 캐시, 정규화, 품질 분석, 통합 규칙을 서로 분리한다. 센터 집계는 명시적으로 승인된 규칙이 없으면 실패하도록 한다.

**기술 스택:** Python 3.13.5, pandas 2.2.3, requests 2.32.3, pytest 8.3.4, pyarrow 19.0.0

**설계 문서:** `docs/superpowers/specs/2026-11-11-api-quality-design.md`

## 전역 제약

- 기본 실험 종료일은 2026-09-30이다.
- 서울 API 페이지 크기는 최대 1,000건이다.
- 인증키와 비밀 URL은 파일·로그·예외에 남기지 않는다.
- 원본 응답과 원본 CSV는 수정하지 않는다.
- COD와 TOC, 유입수와 방류수, 1차와 2차 처리량을 합치지 않는다.
- 미검증 센터 통합 규칙은 실행하지 않는다.

## 검토 중점

- HTTP 200 안의 API 오류 코드가 성공으로 처리되지 않는가?
- 마지막 페이지 재개 시 중복 행이 생기지 않는가?
- 0과 결측, 강수 0mm와 미관측이 구분되는가?
- 처리장 경계와 유입/방류 계통이 집계 과정에서 사라지지 않는가?
- CP949 열 이름과 수치 변환 오류가 조용히 누락되지 않는가?

---

### 작업 1: 설정과 스키마

**파일:** `src/data/settings.py`, `src/preprocessing/schema.py`, `tests/test_schema.py`

**인터페이스:** `load_settings() -> Settings`, `normalize_quality_columns(df)`, `normalize_flow_columns(df)`

- [ ] 요구 열 누락과 COD/TOC·TP 매핑을 검증하는 실패 테스트 작성
- [ ] 실패 확인 후 최소 설정·스키마 구현
- [ ] 테스트 통과 확인

### 작업 2: API 전송과 페이지네이션

**파일:** `src/data/api_client.py`, `src/data/seoul_api.py`, `tests/test_seoul_api.py`

**인터페이스:** `build_retrying_session()`, `SeoulApiClient.iter_rows(service_name)`

- [ ] 1,000행 페이지, 총건수 종료, 오류 코드, 키 비노출 테스트 작성
- [ ] 실패 확인 후 최소 구현
- [ ] 테스트 통과 확인

### 작업 3: 원본 캐시와 재개

**파일:** `src/data/cache.py`, `tests/test_cache.py`

**인터페이스:** `RawPageCache.save_page()`, `load_checkpoint()`, `write_manifest()`

- [ ] 원자적 저장·기존 페이지 재사용·체크포인트 테스트 작성
- [ ] 실패 확인 후 최소 구현
- [ ] 테스트 통과 확인

### 작업 4: CSV 품질 분석

**파일:** `src/preprocessing/quality.py`, `scripts/analyze_csv.py`, `tests/test_quality.py`

**인터페이스:** `read_source_csv(path, encoding='cp949')`, `build_quality_summary(df, ...)`, `detect_outlier_candidates(df, ...)`

- [ ] 결측률, 연속 결측, 0, 중복, 음수, IQR·MAD 후보 테스트 작성
- [ ] 실패 확인 후 최소 구현
- [ ] Markdown·CSV 출력 테스트와 전체 테스트 통과 확인

### 작업 5: 통합 규칙 안전장치

**파일:** `src/preprocessing/integration.py`, `configs/integration_rules.yaml`, `tests/test_integration.py`

**인터페이스:** `validate_rule(rule)`, `aggregate_center_daily(df, rule)`

- [ ] `보류` 규칙 실행 거부와 중복 처리량 합산 방지 테스트 작성
- [ ] 검증된 `대표값`·`합계`·`유량가중평균`만 허용하는 최소 구현
- [ ] 테스트 통과 확인

### 작업 6: 문서와 실행 검증

**파일:** `README.md`, `docs/*.md`, `.env.example`, `.gitignore`, `requirements.txt`

- [ ] API 키 없이 실행되는 테스트와 CSV 부재 안내 명령 확인
- [ ] `pytest -q` 전체 통과 확인
- [ ] 실제 CSV와 인증키가 필요한 남은 작업을 보고서에 기록
