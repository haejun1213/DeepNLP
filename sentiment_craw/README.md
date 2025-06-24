# 도서 리뷰 감성 분석 대시보드

**교보문고에서 원하는 도서를 검색하고, 해당 도서의 온라인 리뷰를 크롤링하여 긍정/중립/부정 감성을 분석하고 시각화하는 Streamlit 웹 애플리케이션입니다.**

---

## 주요 기능

### 도서 검색
- 사용자가 입력한 키워드를 기반으로 Selenium을 사용하여 교보문고의 동적 검색 결과 페이지에 접속
- BeautifulSoup으로 HTML 구조를 파싱하여 제목, 저자, 출판사 및 상세 URL을 추출

![image](https://github.com/user-attachments/assets/f79669dd-7160-4ae7-9cd8-96c571e67e42)


### 리뷰 실시간 크롤링
- WebDriverWait과 `staleness_of` 조건을 활용해 동적 페이지네이션 처리
- ‘다음’ 버튼을 끝까지 클릭하여 모든 리뷰(텍스트, 평점, 작성자, 작성일)를 수집

![image](https://github.com/user-attachments/assets/13d97c18-8195-4faa-aa9e-a6426f2b29e7)


### AI 기반 감성 분석
- Hugging Face의 `klue/bert-base` 모델을 1만 개의 한국어 리뷰로 Fine-tuning한 커스텀 모델 사용
- 각 리뷰를 긍정/중립/부정으로 분류하고, 클래스 확률 계산

![image](https://github.com/user-attachments/assets/b19d0896-63b0-4633-88ae-8af8587d5aac)

![image](https://github.com/user-attachments/assets/7a16a6ee-7d3b-43f1-bd03-d356945e0eb0)
![image](https://github.com/user-attachments/assets/56d5bf95-ad6e-4afc-9f21-b2fa0020e23d)
![image](https://github.com/user-attachments/assets/86ea63c9-c90c-4e89-bcb2-990c964a756e)


### 데이터 시각화
- **감성 분포**: Seaborn 막대그래프
  ![image](https://github.com/user-attachments/assets/43aab7cd-01d0-4f5a-b20f-95dc25c5d296)

- **워드클라우드**: WordCloud 라이브러리 활용
  ![image](https://github.com/user-attachments/assets/eb33538e-adf4-4ce6-bb48-f78b04094c1d)

- **평점 분포**: Seaborn 히스토그램
  ![image](https://github.com/user-attachments/assets/a1346e6b-f36f-43e1-a901-8cc286b46c75)


### 결과 다운로드
- Pandas DataFrame으로 분석 결과 정리
- CSV 파일로 사용자에게 제공 (리뷰 내용, 평점, 예측 감성 등 포함)


### CSV 파일 기반 시각화
- csv 파일 저장해 두면 데이터 시각화 가능
  ![image](https://github.com/user-attachments/assets/0e1313d6-b056-457d-838c-3a0a64ca5eff)

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
├── labeled_reviews_40k.csv
├── main_app.py
├── train_model.ipynb
├── requirements.txt
└── README.md
```
---

## 동작 원리

### 1. 감성 분석 모델 학습 (`train_model.ipynb`)

#### 데이터 준비
- `labeled_reviews_30k.csv`: 30,000개의 한국어 리뷰 (긍정/중립/부정 → 2/1/0) , 알라딘 도서에서 크롤링 해온 데이터
- `train_test_split`: 학습 80% / 검증 20%
- LLM(gemma) 사용하여 라벨링
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

![image](https://github.com/user-attachments/assets/3079aeb4-16b2-4aad-94a6-b5fd923a0e98)


#### 평가 및 저장
- `classification_report`로 정확도/정밀도/재현율 확인
- 모델 및 토크나이저 파일 저장

---

### 2. Streamlit 웹 애플리케이션 (`main_app.py`, `pages/Analyze_Reviews.py`,  `pages/Visualize_CSV.py`)

#### 도서 검색 (`main_app.py`)
- 키워드 기반 검색 페이지 URL 생성
- Selenium + BeautifulSoup으로 도서 목록 크롤링 및 표시

#### 리뷰 분석 (`Analyze_Reviews.py`)
- 도서 상세 페이지 진입 후 리뷰 페이지네이션 순회 크롤링
- WebDriverWait으로 안정적인 페이지 전환 감지
- 크롤링한 리뷰 → `predict_sentiment_tf()` → 감성 예측
- 결과를 Pandas로 정리, 시각화 결과 표시

#### CSV 기반 데이터 시각화 (`pages/Visualize_CSV.py`)
- 저장해둔 csv로 시각화 가능

#### 웹 크롤링 및 HTML 분석 상세
- 이 프로젝트의 핵심 데이터 수집은 `Selenium`과 `BeautifulSoup`을 통해 이루어집니다.

* **기술 스택**:
    * **`Selenium`**: 동적 웹 페이지 제어를 담당합니다. JavaScript가 실행되어야만 내용이 보이는 검색 결과 페이지나 리뷰의 페이지네이션(페이지 넘김)과 같이 사용자와의 상호작용이 필요한 부분을 자동화합니다.
    * **`BeautifulSoup`**: `Selenium`으로 가져온 정적인 HTML 소스 코드를 파싱(분석)하여 원하는 데이터를 쉽게 추출하는 역할을 합니다. CSS 선택자(Selector)를 통해 필요한 정보(도서명, 저자, 리뷰 내용 등)에 정확하게 접근합니다.

* **크롤링 프로세스**:
    1.  **도서 검색**: 사용자가 입력한 키워드로 교보문고의 검색 URL을 생성하고 접속합니다. `Selenium`의 `WebDriverWait`를 사용해 검색 결과 목록(`ul.prod_list`)이 완전히 로드될 때까지 대기하여 안정적으로 데이터를 확보합니다.
    2.  **리뷰 페이지 이동**: 사용자가 선택한 도서의 상세 URL로 이동합니다.
    3.  **동적 페이지네이션 처리**: 리뷰 섹션은 여러 페이지로 나뉘어 있으며, '다음' 버튼을 눌러야만 새로운 리뷰가 로드되는 동적 구조를 가지고 있습니다. 이 문제를 해결하기 위해 다음과 같은 로직을 사용합니다.
        * `while True` 루프를 통해 '다음' 버튼이 비활성화될 때까지 반복적으로 페이지를 탐색합니다.
        * 현재 페이지의 첫 번째 리뷰 요소를 변수에 저장합니다.
        * `Selenium`으로 '다음' 버튼을 클릭합니다.
        * `WebDriverWait(driver, 15).until(EC.staleness_of(first_review_element_on_page))` 코드를 실행합니다. 이 코드는 **페이지 전환을 안정적으로 감지하는 핵심 로직**으로, 이전에 저장해둔 첫 번째 리뷰 요소가 페이지에서 사라질 때(stale)까지 명시적으로 기다립니다. 이를 통해 새로운 리뷰 목록이 완전히 로드되었음을 보장하고 데이터 누락을 방지합니다.
    4.  **데이터 추출**: 새로운 리뷰 목록이 로드되면, `driver.page_source`를 `BeautifulSoup`에 전달하여 HTML을 파싱하고, 각 리뷰 아이템(`div.comment_item`)에서 닉네임, 평점, 리뷰 본문 등의 데이터를 추출하여 리스트에 저장합니다.
  
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

---

# 향후 개선 사항
## 비동기 크롤링: asyncio, aiohttp 활용하여 속도 향상

## 모델 경량화: DistilBERT, MobileBERT 등 적용
