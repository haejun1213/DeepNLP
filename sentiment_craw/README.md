# 도서 리뷰 감성 분석 대시보드

**교보문고에서 원하는 도서를 검색하고, 해당 도서의 온라인 리뷰를 실시간으로 크롤링하여 긍정/중립/부정 감성을 분석하고 시각화하는 Streamlit 웹 애플리케이션입니다.**

---

## 주요 기능

### 도서 검색
- 사용자가 입력한 키워드를 기반으로 Selenium을 사용하여 교보문고의 동적 검색 결과 페이지에 접속
- BeautifulSoup으로 HTML 구조를 파싱하여 제목, 저자, 출판사 및 상세 URL을 추출

### 리뷰 실시간 크롤링
- WebDriverWait과 `staleness_of` 조건을 활용해 동적 페이지네이션 처리
- ‘다음’ 버튼을 끝까지 클릭하여 모든 리뷰(텍스트, 평점, 작성자, 작성일)를 수집

### AI 기반 감성 분석
- Hugging Face의 `klue/bert-base` 모델을 1만 개의 한국어 리뷰로 Fine-tuning한 커스텀 모델 사용
- 각 리뷰를 긍정/중립/부정으로 분류하고, 클래스 확률 계산

### 데이터 시각화
- **감성 분포**: Seaborn 막대그래프
- **워드클라우드**: WordCloud 라이브러리 활용
- **평점 분포**: Seaborn 히스토그램

### 결과 다운로드
- Pandas DataFrame으로 분석 결과 정리
- CSV 파일로 사용자에게 제공 (리뷰 내용, 평점, 예측 감성 등 포함)

---

## 프로젝트 구조
```
.
├── fonts/
│ └── NanumGothic.ttf
├── my_custom_bert_sentiment_model/
│ ├── tf_model.h5
│ ├── config.json
│ ├── special_tokens_map.json
│ ├── tokenizer_config.json
│ └── vocab.txt
├── pages/
│ └── Analyze_Reviews.py
├── labeled_reviews_10k.csv
├── main_app.py
├── train_model.py
├── requirements.txt
└── README.md
```
---

## 동작 원리

### 1. 감성 분석 모델 학습 (`train_model.py`)

#### 데이터 준비
- `labeled_reviews_10k.csv`: 10,000개의 한국어 리뷰 (긍정/중립/부정 → 2/1/0)
- `train_test_split`: 학습 80% / 검증 20%

#### 모델 아키텍처
- `klue/bert-base` 기반 `TFBertModel` 사용
- Dropout + Dense로 구성된 커스텀 분류기 구축

#### 토큰화
- `BertTokenizer`로 input_ids, attention_mask 생성
- `MAX_LEN=128`로 패딩/잘림 처리

#### 모델 컴파일 및 학습
- 옵티마이저: `Adam(lr=2e-5)`
- 손실함수: `SparseCategoricalCrossentropy`
- 배치: 16 / 에폭: 5

#### 평가 및 저장
- `classification_report`로 정확도/정밀도/재현율 확인
- 모델 및 토크나이저 파일 저장

---

### 2. Streamlit 웹 애플리케이션 (`main_app.py`, `pages/Analyze_Reviews.py`)

#### 도서 검색 (`main_app.py`)
- 키워드 기반 검색 페이지 URL 생성
- Selenium + BeautifulSoup으로 도서 목록 크롤링 및 표시

#### 리뷰 분석 (`Analyze_Reviews.py`)
- 도서 상세 페이지 진입 후 리뷰 페이지네이션 순회 크롤링
- WebDriverWait으로 안정적인 페이지 전환 감지
- 크롤링한 리뷰 → `predict_sentiment_tf()` → 감성 예측
- 결과를 Pandas로 정리, 시각화 결과 표시

---

## 설치 및 실행 방법

### 1. 사전 요구사항
- Python 3.8 이상
- Google Chrome + `chromedriver` (자동 설치됨)

### 2. 설치 과정

#### Git 리포지토리 클론

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

💻 가상 환경 생성 (선택 사항)

```
python -m venv venv

```
# Windows

```
venv\Scripts\activate
```
# macOS / Linux

```
source venv/bin/activate
```

📦 필수 패키지 설치

```
pip install -r requirements.txt
```
my_custom_bert_sentiment_model/ 디렉토리에 학습된 모델(.h5 등) 복사

직접 학습 시: train_model.py 실행

폰트 설치 (워드클라우드용)
fonts/NanumGothic.ttf 위치

시스템 폰트(맑은 고딕, AppleGothic)가 없을 경우 필요

3. 실행
```
streamlit run main_app.py
```
브라우저에서 http://localhost:8501 접속

주요 의존성
streamlit

pandas

selenium

webdriver-manager

beautifulsoup4

tensorflow

transformers

scikit-learn

matplotlib

seaborn

wordcloud

전체 목록은 requirements.txt 참조

향후 개선 사항
비동기 크롤링: asyncio, aiohttp 활용하여 속도 향상

모델 경량화: DistilBERT, MobileBERT 등 적용

클라우드 배포: Streamlit Cloud / AWS / GCP 지원

DB 연동: 분석 결과를 DB에 저장, 이력 관리 기능 추가

고급 시각화: 감성 변화 추이, 감성별 키워드 분석 등
