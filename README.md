# 회원 프로필 사진 인증 자동화를 위한 VISION 모델 개발

### 개발 기간 
2022.03 ~ 2023.10 ( 21개월 )

### 핵심성과
9가지 객체 인 모델을 도입하여 회원의 사진 인증까지 걸리는 시간을 단축하고 프로필 인증 담당자의 업무 95% 감소

### 담당 역할 :
1. 이미지 데이터 수집 및 전처리
2. VISION 모델 개발
3. 모델 배포(API 개발)

### 프로세스 설명 :
1. 회원 프로필 사진 업로드 시 앱 벡엔드 서버에서 YOLO api로 요청
2. 요청이 오면 해당 회원의 사진을 서버에서 불어와 객체 인식
3. 객체의 라벨과 신뢰도에 따라 자동인증, 자동거부, 대기 상태 중 하나로 응답
4. 앱 벡인드 서버에서 자동 처리 및 대기 상태인 사진만 CS팀에서 처리

### 개발환경
- Python 3.10
- torch 2.3.0
- CUDA 11.8

### 코드 설명
- mask_filtering - FastAPI로 구동되는 YOLO 모델 메인 파일
- mask_functions - YOLO 모델 로드와 객체 인식을 위한 함수
- mask_start - 서버 상에서 FastAPI를 실행 시 사용하는 파일(서버에서는 데몬 구동)
- conf 폴더 - API를 구동하는데 공통으로 사용되는 구성과 로그 설정
- models, utils 폴더 - YOLO모델을 구동시기키위한 폴더
