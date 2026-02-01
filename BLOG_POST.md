DocuAgent 프로젝트 소개

(대표 이미지 첨부: assets/docuagent-onepage.svg)
이미지 캡션: DocuAgent One Page Overview — 문서 분석부터 교육 콘텐츠 생성까지 한눈에

문제 정의
문서 기반 업무나 교육 자료를 만들 때 같은 문제가 반복됩니다. 문서를 사람이 직접 읽고 구조를 정리해야 하고, 필요한 정보만 골라내는 데 시간이 오래 걸립니다. 요약이나 퀴즈 같은 교육 자료를 매번 새로 만들어야 하며, 결과를 LMS나 문서 도구로 옮기는 과정도 번거롭습니다. 입문자 입장에서는 자동화 흐름 자체가 어렵게 느껴지는 것도 큰 장벽입니다.

해결 방안
DocuAgent는 문서를 올리면 구조화, 핵심 정보 추출, 요약, 교육 콘텐츠 생성, 문서 기반 질의응답까지 한 번에 처리합니다. 내부 흐름은 다음과 같습니다.
1) 문서 파싱으로 구조화 텍스트 생성
2) 문서 유형 분석 후 추출 스키마 자동 생성
3) 스키마 기반 핵심 정보 JSON 추출
4) 요약과 교육 콘텐츠(학습 목표, 퀴즈 등) 생성
5) 문서 기반 질의응답 제공
6) 결과를 Markdown/HTML/PDF/SCORM/IMS로 내보내기

기대효과
- 문서 분석 시간 절감: 수작업 정리 시간을 크게 줄임
- 교육 콘텐츠 제작 자동화: 기본 구조를 자동 생성해 제작 부담 감소
- 내보내기 자동화: LMS, Notion 등에 바로 적용 가능
- 입문자 접근성 향상: 업로드 → 확인 → 내보내기 흐름만 따라가면 됨

핵심 기술
- Upstage Document Parse: 문서를 Markdown으로 구조화
- Upstage Solar: 문서 유형 분석, 스키마 생성, 요약 및 질의응답, 교육 콘텐츠 생성
- Upstage Information Extract: 스키마 기반 정보 추출
- FastAPI 기반 백엔드, HTML/CSS/JS 프론트엔드
- SCORM 1.2/2004, IMS CC 1.1/1.3 패키지 생성

입문자용 설명
문서를 업로드하면 먼저 Document Parse가 텍스트 구조를 만들고, Solar가 문서 유형을 분석해 어떤 정보를 뽑을지 스키마를 자동으로 만듭니다. 이후 Information Extract가 스키마에 맞춰 데이터를 추출하고, Solar가 요약과 교육 콘텐츠를 생성합니다. 마지막으로 결과를 원하는 형식으로 저장하거나 LMS에 가져올 수 있습니다.

빠른 시작
1) 설치
python3 -m pip install -r requirements.txt

2) API 키 설정
export UPSTAGE_API_KEY="your_api_key_here"

3) 실행
python3 main.py
브라우저에서 http://localhost:8000 접속

사용 방법
1. 문서 업로드
2. 학습자 수준/학습 목표 선택
3. LOM 태그 입력(선택)
4. 결과 확인
5. Markdown/HTML/PDF/SCORM/IMS로 내보내기
6. 문서 기반 질의응답

데모 문서
학습용 흐름을 자연스럽게 보여주려면 교육 자료 형식의 문서가 좋습니다.
- /Users/kim/Downloads/docuagent/assets/demo-learning-ko.pdf

마무리
DocuAgent는 문서 분석부터 교육 콘텐츠 제작, LMS 연동까지 연결하는 실용적인 파이프라인입니다. 실제 업무나 교육 현장에 바로 적용할 수 있도록 구성했습니다.
