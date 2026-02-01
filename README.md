DocuAgent

요약
DocuAgent는 문서를 업로드하면 구조화, 핵심 정보 추출, 요약, 교육 콘텐츠 생성, 문서 기반 질의응답까지 한 번에 제공하는 문서 분석 서비스입니다.

문서
- 한국어: README.ko.md
- English: README.en.md

빠른 시작
1) 설치
python3 -m pip install -r requirements.txt

2) API 키 설정
export UPSTAGE_API_KEY="your_api_key_here"

3) 실행
python3 main.py
브라우저에서 http://localhost:8000 접속

프로젝트 구조
- main.py: 백엔드 및 Upstage API 연동
- index.html: 프론트엔드 UI
- assets/: 이미지 및 문서 리소스
- requirements.txt: 의존성

데모 문서
- assets/demo-learning-ko.pdf
- assets/demo-learning.pdf
