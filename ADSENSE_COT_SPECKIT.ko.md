# Upstage DocuAgent AdSense COT + SPECKIT 운영문서

Last updated: 2026-02-18

## 1) COT 방식 점검 로그 (Chain of Thought 운영 버전)

이 문서는 심사 통과 가능성을 높이기 위한 실행 점검 기록용 체크리스트입니다.

### COT-01 콘텐츠 가치
- [ ] 메인 화면에서 서비스 목적이 5초 내 이해된다.
- [ ] 핵심 기능이 광고보다 먼저 보이고, 광고 없이도 서비스 가치가 성립한다.
- [ ] About/Privacy/Terms/Contact/Compliance 페이지가 실제 운영 설명을 포함한다.

### COT-02 정책/신뢰
- [ ] 연락 채널이 가짜 도메인(.local 등)이 아니다.
- [ ] 정책 링크가 홈 또는 주요 UI에서 바로 접근된다.
- [ ] robots.txt, sitemap.xml, ads.txt가 공개 루트에서 접근된다.

### COT-03 광고 배치
- [ ] Sponsored/광고 라벨이 명시된다.
- [ ] 광고가 CTA 버튼/핵심 입력 요소와 붙어 있지 않다.
- [ ] 유효한 publisher/slot 값이 없으면 광고 요청을 보내지 않는다.

### COT-04 기술 안정성
- [ ] 빌드 성공
- [ ] 런타임 에러 없음
- [ ] 페이지 로딩 중 광고 스크립트 실패 시에도 UI가 깨지지 않음

## 2) SPECKIT 방식 사양서

### S (Scope)
- 대상: Upstage DocuAgent 웹 배포 영역
- 웹 루트: `.`

### P (Policy)
- Google 정책 위반 가능 요소(오해 유발 배치, 클릭 유도 문구, 불명확 연락처) 제거
- 정책 페이지를 탐색 가능한 위치에 유지

### E (Execution)
- 배포 전: `tools/release_ops.sh check` 실행
- 값 반영: `tools/release_ops.sh apply-adsense <ca-pub-...> <slot-id>`

### C (Criteria)
- PASS 조건:
  - 필수 파일: ads.txt / robots.txt / sitemap.xml
  - 필수 정책: about/privacy/terms/contact/compliance
  - 가짜 연락처 미검출
  - Sponsored 라벨 검출

### K (Keep)
- 릴리즈마다 동일 체크 반복
- 정책/문의 페이지 실제 운영 내용으로 주기 업데이트

### I (Improve)
- 심사 피드백 발생 시 원인-수정-재검증 로그를 본 문서 하단에 누적

### T (Trace)
- 변경 추적: git commit + 배포 URL + 검사 결과를 함께 기록

## 3) 배포 전 빠른 명령

- `bash tools/release_ops.sh check`
- 실값 반영 후: `bash tools/release_ops.sh apply-adsense ca-pub-XXXXXXXXXXXXXXXX 1234567890`
- 재검증: `bash tools/release_ops.sh check`

## 4) 리뷰어 관점 메모

- 광고는 보조 슬롯이며, 서비스 핵심 기능과 분리되어 있습니다.
- 정책 페이지와 문의 채널은 공개되어 있고 탐색 가능합니다.
- placeholder 값 환경에서는 광고 요청이 차단되도록 구성했습니다.
