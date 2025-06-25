# 도서 리뷰 감성 분석

본 문서는 **교보문고 도서 리뷰 데이터**를 활용한 **한국어 감성 분석 모델 개발** 프로젝트의 기술적인 측면을 상세히 기록한 보고서입니다. 프로젝트는 크게 웹 크롤링을 통한 데이터 수집과 BERT 기반 모델 학습 및 평가 두 단계로 구성됩니다. 이 보고서는 모델의 설계, 학습 과정, 성능 평가에 대한 심층적인 분석을 제공하여 구현된 기술 스택의 역량을 증명하는 데 목적이 있습니다.


## 목차

- 1. 프로젝트 전체 흐름도

- 2. 기술 스택

- 3. 동적 웹 크롤링 아키텍처

  - 3.1. 교보문고 웹 페이지 구조 분석

  - 3.2. 크롤링 핵심 전략: WebDriverWait과 staleness_of

- 4. 감성 분석 모델

  - 4.1. 데이터 준비 및 분석

  - 4.2. 모델 아키텍처

  - 4.3. 모델 학습 과정 분석

  - 4.4. 모델 성능 평가

  - 4.5. 추론 및 모델 영속성

- 5. 환경 설정 및 의존성

- 6. 향후 개선 사항

## 1. 프로젝트 전체 흐름도

![image](https://github.com/user-attachments/assets/b85a72bc-ca88-480e-9b32-2cbea45bc913)

## 2. 기술 스택

| 분야        | 기술                               | 설명                                         |
|-------------|------------------------------------|----------------------------------------------|
| 데이터 수집 | Selenium, BeautifulSoup4           | 동적 웹 페이지의 콘텐츠 수집 및 HTML 파싱   |
| 레이블링    | Google Gemma                       | LLM 기반 자동 감성 레이블 생성              |
| 모델링      | TensorFlow, Huggingface Transformers | BERT 기반 분류 모델 파인튜닝                |
| 데이터 처리 | Pandas, NumPy                      | 데이터 로딩, 정제, 분석                     |
| 성능 평가   | Scikit-learn                       | classification_report 기반 정량적 분석      |

## 3. 동적 웹 크롤링 아키텍처

### 3.1 교보문고 웹 페이지 구조 분석
- 동적 웹 페이지 (JavaScript + AJAX 기반 비동기 로딩)
- Selenium을 통해 전체 DOM 확보 후 BeautifulSoup으로 파싱

### 3.2 크롤링 핵심 전략: WebDriverWait과 staleness_of
- time.sleep() 대신 명시적 대기 방식으로 안정성 확보

```
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)
driver.get("https://product.kyobobook.co.kr/detail/...")

while True:
    try:
        first_review = driver.find_element(By.CSS_SELECTOR, "div.comment_item")
        next_btn = driver.find_element(By.CSS_SELECTOR, "a.btn_next")
        next_btn.click()
        WebDriverWait(driver, 15).until(EC.staleness_of(first_review))
    except:
        break
```

## 4. 감성 분석 모델

### 4.1 데이터 준비 및 분석
- 약 30만 건 리뷰에 LLM(Gemma) 기반 자동 레이블링
- 클래스 비율: 긍정 65%, 중립 22.9%, 부정 11.8%
```
label_mapping = {'부정': 0, '중립': 1, '긍정': 2}
df['label'] = df['label'].map(label_mapping)
```

### 4.2 모델 아키텍처
- Huggingface klue/bert-base 사용
- Pooler Output 기반 커스텀 분류기 정의
```
class CustomBertForSequenceClassification(tf.keras.Model):
    def __init__(self, bert_model_core, num_labels):
        super().__init__()
        self.bert = bert_model_core
        self.dropout = Dropout(0.1)
        self.classifier = Dense(num_labels)

    def call(self, inputs, training=False):
        pooled_output = self.bert(**inputs).pooler_output
        pooled_output = self.dropout(pooled_output, training=training)
        return self.classifier(pooled_output)
```

### 4.3 모델 학습 과정 분석
- 학습률: 2e-5 / 배치 크기: 16 / Epoch: 5
- 과적합 징후 명확 → EarlyStopping 필요

### 4.4 모델 성능 평가

![image](https://github.com/user-attachments/assets/264b3157-2638-4f5e-ae07-ede02b093d04)

![image](https://github.com/user-attachments/assets/caa26d00-18f5-4cb2-a3e1-5f974f020361)


- F1-Score: 긍정(0.93), 중립(0.74), 부정(0.78)
- Macro Avg: 0.82 / Weighted Avg: 0.87

### 4.5 추론 및 모델 영속성
```
def predict_sentiment_tf(text, model, tokenizer):
    encoded = tokenizer.encode_plus(text, max_length=128, padding='max_length', return_tensors='tf', truncation=True)
    logits = model(encoded, training=False)
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    return np.argmax(probs), probs
```

### 5. 환경 설정 및 의존성
- 1. 요구사항
     - python >= 3.8
     - Chrome 및 chromedriver

- 2. 가상환경 설정
  ```
  python -m venv venv
  source venv/bin/activate  # macOS/Linux
  venv\Scripts\activate     # Windows
  ```

- 3. 필수 패키지 설치
  ```
  pip install -r requirements.txt
  ```
  

### 6. 향후 개선 사항
- EarlyStopping + ModelCheckpoint로 최적 가중치 저장
- AdamW 옵티마이저 + Learning Rate Warmup 적용
- class_weight, SMOTE 기반 클래스 불균형 처리
- asyncio 기반 크롤링 속도 개선




