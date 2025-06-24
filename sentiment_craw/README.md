# 도서 리뷰 감성 분석 대시보드

교보문고에서 원하는 도서를 검색하고, 해당 도서의 온라인 리뷰를 실시간으로 크롤링하여 긍정/중립/부정 감성을 분석하고 시각화하는 Streamlit 웹 애플리케이션입니다.

![데모 이미지](https://placehold.co/800x400/A1B2CF/FFFFFF?text=애플리케이션+데모+스크린샷)

---

## 주요 기능

* **도서 검색**: 사용자가 입력한 키워드를 기반으로 `Selenium`을 사용하여 교보문고의 동적 검색 결과 페이지에 접속합니다. `BeautifulSoup`으로 HTML 구조를 파싱하여 각 도서의 제목, 저자, 출판사 및 리뷰 페이지로 연결되는 상세 URL을 추출하여 사용자에게 보여줍니다.

* **실시간 리뷰 크롤링**: `Selenium`의 `WebDriverWait`와 `staleness_of` 조건을 활용하여, 동적으로 로드되는 리뷰 페이지네이션을 안정적으로 처리합니다. '다음' 버튼을 끝까지 클릭하며 모든 페이지의 리뷰 텍스트, 평점, 작성자, 작성 날짜 데이터를 수집합니다.

* **AI 기반 감성 분석**: Hugging Face의 `klue/bert-base` 모델을 1만 개의 라벨링된 한국어 리뷰 데이터로 미세 조정한 커스텀 모델을 사용합니다. `TensorFlow`와 `Transformers` 라이브러리를 통해 크롤링한 각 리뷰 텍스트를 **긍정, 중립, 부정** 세 가지 클래스로 분류하고, 각 클래스에 대한 확률을 계산합니다.

* **데이터 시각화**: `Matplotlib`, `Seaborn`, `WordCloud` 라이브러리를 활용하여 분석 결과를 직관적으로 시각화합니다.
    * **감성 분포**: `Seaborn` 막대그래프를 사용하여 예측된 감성(긍정/중립/부정)의 전체적인 비율을 한눈에 파악할 수 있도록 합니다.
    * **워드클라우드**: `WordCloud`를 이용해 전체 리뷰에서 자주 언급된 핵심 단어들을 시각적으로 강조하여 보여줍니다.
    * **평점 분포**: `Seaborn` 히스토그램을 통해 사용자들이 부여한 평점의 분포를 시각화하여 리뷰의 전반적인 만족도를 파악할 수 있게 합니다.

* **결과 다운로드**: 크롤링하고 분석한 모든 리뷰 데이터(리뷰 내용, 평점, 예측 감성, 감성별 확률 등)를 `Pandas` DataFrame으로 정리한 뒤, 사용자가 보관하고 추가 분석할 수 있도록 CSV 파일 형태로 제공합니다.

---

## 프로젝트 구조


.
├── fonts/
│   └── NanumGothic.ttf         # (옵션) 워드클라우드용 폰트 파일
├── my_custom_bert_sentiment_model/
│   ├── tf_model.h5             # 학습된 모델 가중치
│   ├── config.json             # 모델 설정 파일
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt               # 토크나이저 어휘 파일
├── pages/
│   └── Analyze_Reviews.py      # 리뷰 분석 및 시각화 페이지
├── labeled_reviews_10k.csv     # (학습용) 감성 라벨링된 리뷰 데이터
├── main_app.py                 # 메인 애플리케이션 (도서 검색)
├── train_model.py              # 감성 분석 모델 학습 스크립트
├── requirements.txt            # 프로젝트 의존성 파일
└── README.md                   # 프로젝트 설명 파일


---

## 동작 원리

### 1. 감성 분석 모델 학습 (`train_model.py`)

1. **데이터 준비**: `labeled_reviews_10k.csv` 파일에 포함된 1만 개의 한국어 리뷰 데이터를 로드합니다. 각 리뷰는 '긍정', '중립', '부정' 라벨을 가집니다.
2. **모델 아키텍처**: Hugging Face의 `klue/bert-base` 모델을 기반으로 한 `CustomBertForSequenceClassification` 커스텀 모델을 정의합니다. BERT 모델 위에 Dropout 레이어와 분류를 위한 Dense 레이어를 추가하여 구성합니다.
3. **토크나이저**: `klue/bert-base` 모델의 `BertTokenizer`를 사용하여 텍스트 데이터를 모델이 이해할 수 있는 토큰으로 변환합니다.
4. **학습 및 평가**: 준비된 데이터셋으로 모델을 5 epoch 동안 미세 조정(Fine-tuning)합니다. 학습 후, 검증 데이터셋으로 성능을 평가하고 `classification_report`를 통해 정확도, 정밀도, 재현율을 확인합니다.
5. **모델 저장**: 학습이 완료된 모델의 가중치(`tf_model.h5`)와 토크나이저 파일을 `my_custom_bert_sentiment_model/` 디렉토리에 저장하여 애플리케이션에서 사용할 수 있도록 합니다.

### 2. Streamlit 웹 애플리케이션 (`main_app.py` & `Analyze_Reviews.py`)

1. **도서 검색 (`main_app.py`)**:
   - 사용자가 입력한 키워드로 교보문고 검색 페이지 URL을 생성합니다.
   - `Selenium`을 사용하여 해당 페이지에 접속하고, 동적으로 렌더링된 검색 결과를 크롤링합니다.
   - `BeautifulSoup`으로 HTML을 파싱하여 도서명, 저자, 출판사, 상세 페이지 URL 등의 정보를 추출하여 표 형태로 보여줍니다.

2. **리뷰 분석 (`Analyze_Reviews.py`)**:
   - 사용자가 선택한 도서의 상세 페이지로 `Selenium`을 통해 이동합니다.
   - 리뷰 섹션의 첫 페이지를 로드한 후, '다음' 버튼이 비활성화될 때까지 클릭을 반복하며 모든 리뷰 페이지를 크롤링합니다.
     - **페이지네이션 처리**: 페이지가 동적으로 로드되므로, `WebDriverWait`의 `staleness_of` 조건을 사용하여 이전 페이지의 요소가 사라지는 것을 감지하고 다음 페이지가 완전히 로드될 때까지 안정적으로 대기합니다.
   - 수집된 각 리뷰 텍스트를 `predict_sentiment_tf` 함수에 전달합니다. 이 함수는 사전에 학습된 `my_custom_bert_sentiment_model`을 로드하여 리뷰의 감성을 예측합니다.
   - 예측된 감성, 원본 리뷰 내용, 평점 등을 `pandas` DataFrame으로 정리합니다.
   - `Matplotlib`, `Seaborn`, `WordCloud`를 사용하여 분석 결과를 시각화하여 대시보드에 표시합니다.

---

## 설치 및 실행 방법

### 1. 사전 요구사항

- Python 3.8 이상
- Google Chrome 브라우저 및 `chromedriver` (본 프로젝트는 `webdriver-manager`를 사용하여 자동으로 설치됩니다)

### 2. 설치 과정

1.  **Git 리포지토리 복제:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **가상 환경 생성 및 활성화 (권장):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS / Linux
    source venv/bin/activate
    ```

3.  **필수 라이브러리 설치:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **학습된 모델 다운로드:**
    - `my_custom_bert_sentiment_model/` 디렉토리에 미리 학습된 모델 파일들을 위치시킵니다. (만약 직접 학습하려면 `train_model.py`를 실행하세요.)

5.  **폰트 파일 준비 (옵션):**
    - 워드클라우드에 사용할 한글 폰트가 없다면, `fonts/` 디렉토리를 생성하고 `NanumGothic.ttf`와 같은 폰트 파일을 넣어주세요.
    - 파일이 없는 경우, 코드가 자동으로 시스템에 설치된 '맑은 고딕'이나 'AppleGothic'을 탐색합니다.

### 3. 실행 방법

아래 명령어를 터미널에 입력하여 Streamlit 애플리케이션을 실행합니다.

```bash
streamlit run main_app.py

웹 브라우저에서 http://localhost:8501 주소로 접속하여 애플리케이션을 사용할 수 있습니다.

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

전체 목록은 requirements.txt 파일을 참고해주세요.

향후 개선 사항
비동기 크롤링: asyncio와 aiohttp를 사용하여 크롤링 속도를 개선.

모델 경량화: DistilBERT나 MobileBERT 같은 경량화된 모델을 사용하여 예측 속도 및 리소스 사용량 최적화.

클라우드 배포: Streamlit Cloud, AWS, GCP 등을 활용하여 웹 애플리케이션을 배포.
