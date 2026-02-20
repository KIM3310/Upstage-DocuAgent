# DocuAgent

## 개요
DocuAgent는 문서를 업로드하면 구조화, 핵심 정보 추출, 요약, 교육 콘텐츠 생성을 수행하고 문서 기반 질의응답을 제공하는 문서 분석 서비스입니다. Upstage Solar, Document Parse, Information Extract를 사용합니다.

## 기능
- 문서 파싱: PDF/이미지 문서를 Markdown으로 변환
- 스키마 자동 생성: 문서 유형 분석 후 추출 스키마 생성
- 정보 추출: 스키마 기반 JSON 추출
- 요약 및 질의응답: 문서 내용 기반 요약/응답
- 교육 콘텐츠 패키지: 학습 목표, 핵심 개념, 퀴즈, 플래시카드, 활동 과제
- 사용자 맞춤: 학습자 수준, 학습 목표, 태그 입력
- 내보내기: Markdown, HTML, PDF, SCORM 1.2/2004, IMS CC 1.1/1.3
- LOM 태그: 사용자 태그와 핵심 개념을 IMS CC 1.3 키워드로 반영
- 추천 태그: 핵심 개념 기반 추천 태그 클릭 적용
- 선택 기능: 텍스트 생성 단계(스키마/요약/챗/교육팩)에 Ollama provider 사용 가능

## 처리 흐름
1) Document Parse
2) 스키마 생성
3) Information Extract
4) 교육 콘텐츠 생성
5) 요약 및 질의응답

## 사용된 Upstage API
- Solar (solar-pro2): 스키마 자동 생성, 요약, 교육 콘텐츠, 질의응답
- Document Parse: 문서 구조화 변환
- Information Extract: 스키마 기반 데이터 추출

## 설치 및 실행
### 1) 의존성 설치
```
pip install -r requirements.txt
```

### 2) 데모 모드 실행 (API 키 없이)
리뷰어가 비용/키 없이도 UI를 끝까지 체험할 수 있도록 데모 모드를 제공합니다.
```
DOCUAGENT_DEMO_MODE=1 python main.py
```

### 3) 라이브 모드 실행 (Upstage API)
```
export UPSTAGE_API_KEY="your_api_key_here"
```
또는 .env 파일 사용
키를 설정하지 않으면 **데모 모드(스텁 출력)**로 동작하여, 외부 API 없이도 UI 흐름을 끝까지 확인할 수 있습니다.

### 4) 라이브 모드 실행 (Ollama provider 사용)
```
export UPSTAGE_API_KEY="your_api_key_here"
export DOCUAGENT_LLM_PROVIDER="ollama"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export OLLAMA_MODEL="llama3.2:latest"
python main.py
```

### 5) 접속
```
python main.py
```
브라우저에서 http://localhost:8000 접속
헬스 체크: http://localhost:8000/healthz

## 사용 방법
1) 문서 업로드
   - 또는 UI에서 샘플 버튼(KO/EN)으로 바로 실행
2) 학습자 수준/학습 목표 선택
3) LOM 태그 입력(선택)
4) (선택) Runtime 패널에서 Upstage 키 / Provider / Ollama 설정 입력
   - 설정은 현재 브라우저 세션에만 적용되며 localStorage에 저장하지 않습니다.
5) 분석 완료 후 결과 확인
6) Markdown/HTML/PDF/SCORM/IMS로 내보내기
7) 문서 기반 질의응답
8) `세션 문서` 패널에서 이전 분석 문서 불러오기/삭제

## 광고(AdSense) 설정
- 퍼블리셔 클라이언트는 `ca-pub-4973160293737562`로 포함되어 있습니다.
- 실제 광고 노출을 위해 `Trust & Policy` 영역의 `AdSense 슬롯 ID`에 승인된 슬롯 번호를 입력하세요.

## 내보내기 형식
- Markdown: Notion 업로드용
- HTML: LMS 업로드용
- PDF: 인쇄 저장
- SCORM 1.2/2004: ZIP 패키지
- IMS CC 1.1/1.3: ZIP 패키지

## LOM 태그
- 사용자 태그 + 핵심 개념을 IMS CC 1.3 키워드에 반영
- 추천 태그 클릭으로 빠른 반영 가능

## 프로젝트 구조
- main.py: FastAPI 백엔드, Upstage/Ollama 연동, 패키지 생성
- index.html: 프론트엔드 UI
- requirements.txt: 의존성
- LMS_IMPORT_CHECKLIST.md: LMS 임포트 체크리스트
- LMS_IMPORT_LOG_TEMPLATE.md: 이슈 기록 템플릿

## 참고
- PDF 입력은 poppler(pdftoppm) 우선, 미설치 환경은 pypdfium2 + Pillow 경로를 사용합니다.
- 데모 모드는 pypdf 기반의 로컬 텍스트 추출을 사용하며 OCR은 수행하지 않습니다.
- Information Extract는 기본적으로 PDF 앞 3페이지를 사용합니다. (`DOCUAGENT_IE_MAX_PAGES`)
- 문서 데이터와 런타임 API 키는 세션 단위 메모리 저장이며, 다른 세션의 `doc_id`에는 접근할 수 없습니다.
- 업로드 파일은 확장자 허용 목록(`DOCUAGENT_ALLOWED_EXTENSIONS`)으로 검증합니다.
- Upstage API 호출은 재시도/타임아웃 설정(`DOCUAGENT_UPSTREAM_RETRY_TOTAL`, `DOCUAGENT_UPSTREAM_TIMEOUT_SEC`)을 따릅니다.
- 세션 단위 요청 제한(`DOCUAGENT_RATE_LIMIT_*`)으로 analyze/chat/runtime API를 보호합니다.
- 프론트 분석 플로우는 비동기 작업 폴링(`/api/analyze/jobs`) 기반이며, 작업 취소 API(`/api/analyze/jobs/{job_id}/cancel`)를 지원합니다.
- 비동기 작업은 세션/전체 동시 작업 상한과 TTL 정리(`DOCUAGENT_MAX_ACTIVE_ANALYSIS_JOBS*`, `DOCUAGENT_ANALYSIS_JOB_TTL_SEC`)를 적용합니다.
- 문서 메모리 보관 기간은 `DOCUAGENT_DOC_TTL_SEC`로 제어합니다.
- 경량 요청 메트릭은 `/api/metrics`에서 확인할 수 있습니다.

## 보안 관련 환경변수
- `DOCUAGENT_CORS_ORIGINS`: 콤마 구분 Origin 허용 목록
- `DOCUAGENT_COOKIE_SECURE`: secure 쿠키 강제 (`1`/`true`)
- `DOCUAGENT_COOKIE_SAMESITE`: `lax` / `strict` / `none`
- `DOCUAGENT_HOST`: 기본값 `127.0.0.1`
- `DOCUAGENT_MAX_ACTIVE_ANALYSIS_JOBS` / `DOCUAGENT_MAX_ACTIVE_ANALYSIS_JOBS_PER_SESSION`: 비동기 분석 동시성 보호
- `DOCUAGENT_ANALYSIS_JOB_TTL_SEC` / `DOCUAGENT_DOC_TTL_SEC`: 메모리 보관 기간
- `DOCUAGENT_METRICS_SAMPLE_SIZE` / `DOCUAGENT_MAX_METRICS_PATHS`: 요청 메트릭 샘플 크기

## 개발 / 테스트
```
pip install -r requirements-dev.txt
pytest -q
```
