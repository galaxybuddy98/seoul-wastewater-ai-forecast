# 도구 목록과 확인 상태

확인일: 2026-11-11

| 도구 | 상태 | 확인 내용 | 방침 |
|---|---|---|---|
| Codex/PowerShell | 사용 가능 | 프로젝트 읽기·테스트 실행 | 주 실행 도구 |
| Serena MCP | 사용 가능 | 현재 세션 노출, 프로젝트 활성화, Python LSP 심볼 분석 성공 | 구조·참조·진단 분석에 사용 |
| forecast-validation | 전역 복사 완료 | 원본은 다른 프로젝트에 유지, SHA-256 일치 확인 | 새 세션에서 자동 발견 재확인 |
| Superpowers | 사용 가능 | 설계, 계획, TDD, 완료 검증 절차 적용 | 단계별 검증에 사용 |
| planning-with-files | 제한적 | 계획 폴더는 존재하나 Windows 샌드박스 잠금으로 갱신 실패 | 기존 폴더 보존, 다음 세션 재확인 |
| pandas/requests/pytest | 사용 가능 | pandas 2.2.3, requests 2.32.3, pytest 8.3.4 | 수집·품질·테스트 |
| Git | 저장소 아님 | 초기화하지 않음 | 사용자 요청 전 원격 작업 금지 |

## Serena 확인 기록

설정 재등록이나 재설치는 하지 않았다. `seoul-wastewater-ai-forecast`를 활성화하고 `seoul_api`, `quality`의 심볼 구조를 분석했다. 동명 파일·패키지 중복을 발견해 패키지 구현으로 정리했다. Serena 언어 서버는 별도 Python 환경에서 pandas·numpy·requests·urllib3를 해석하지 못했지만, 프로젝트 Python의 import와 전체 pytest는 통과했다.

## forecast-validation 적합성

시간순 분할, 예측시점 가용성, 잠긴 테스트, 전처리 fit 범위, 기준모델, 동일 평가행과 슬라이스 보고 원칙은 적합하다. 수요·대여소·행사 표현은 유입수질·센터·기상/운영 변수로 바꿔 해석하며 원본 스킬은 수정하지 않았다. 현재 세션 시작 시 스킬 목록에는 없었으므로 자동 로딩 여부는 새 세션에서 재확인한다.
