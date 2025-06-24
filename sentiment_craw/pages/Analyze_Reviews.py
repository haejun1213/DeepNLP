import streamlit as st
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time

# --- 감성 분석 모델 관련 라이브러리 임포트 ---
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import os

# --- 시각화 관련 라이브러리 임포트 ---
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# --- 한글 폰트 설정을 위한 임포트 및 함수 ---
import matplotlib.font_manager as fm

# 폰트 파일 경로 설정
FONT_PATH = "fonts/NanumGothic.ttf"

# --- 워드클라우드를 위한 폰트 경로 탐색 함수 추가 ---
@st.cache_resource
def get_font_path():
    """
    사용 가능한 한글 폰트 경로를 반환합니다.
    1. 지정된 FONT_PATH (NanumGothic) 확인
    2. Windows의 'Malgun Gothic' 확인
    3. macOS의 'AppleGothic' 확인
    """
    if os.path.exists(FONT_PATH):
        return FONT_PATH

    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    
    if os.name == 'nt': # Windows
        for font_path in font_list:
            if 'malgun' in font_path.lower():
                return font_path
    elif os.name == 'posix': # macOS, Linux
        for font_path in font_list:
             if 'apple' in font_path.lower() and 'gothic' in font_path.lower():
                return font_path

    try:
        if os.name == 'nt':
            return fm.findfont('Malgun Gothic')
        elif os.name == 'posix':
            return fm.findfont('AppleGothic')
    except:
        return None

    return None


@st.cache_resource
def set_korean_font_for_matplotlib(font_path):
    """
    Matplotlib의 기본 폰트를 설정합니다.
    """
    if font_path and os.path.exists(font_path):
        try:
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return "success", f"Matplotlib 한글 폰트 설정을 완료했습니다. (경로: {os.path.basename(font_path)})"
        except Exception as e:
            plt.rcParams['axes.unicode_minus'] = False
            return "error", f"폰트 설정 중 오류 발생: {e}. 기본 폰트로 표시됩니다."
    else:
        plt.rcParams['axes.unicode_minus'] = False
        return "warning", "지정된 한글 폰트를 찾을 수 없어 Matplotlib 그래프의 한글이 깨질 수 있습니다."


# --- Streamlit 페이지 설정 ---
st.set_page_config(
    page_title="리뷰 분석",
    page_icon="📊",
    layout="wide"
)

# --- 폰트 설정 ---
valid_font_path = get_font_path()
font_status_type, font_status_message = set_korean_font_for_matplotlib(valid_font_path)

if font_status_type == "success":
    st.success(font_status_message)
elif font_status_type == "warning":
    st.warning(font_status_message)
else:
    st.error(font_status_message)


st.title("선택 도서 리뷰 분석")

# --- Selenium WebDriver 설정 ---
@st.cache_resource
def get_driver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception as e:
        st.error(f"웹 드라이버 초기화 중 오류 발생: {e}")
        st.stop()

driver = get_driver()

# --- 감성 분석 모델 클래스 정의 ---
class CustomBertForSequenceClassification(tf.keras.Model):
    def __init__(self, bert_model_core, num_labels, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.bert = bert_model_core
        self.dropout = Dropout(dropout_rate)
        self.classifier = Dense(num_labels, name="classifier")

    def call(self, inputs, training=False):
        outputs = self.bert(inputs['input_ids'], attention_mask=inputs['attention_mask'], training=training)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        return logits

# --- 감성 분석 모델 및 토크나이저 로드 ---
@st.cache_resource
def load_sentiment_model(model_save_path):
    MODEL_NAME = 'klue/bert-base'
    MAX_LEN = 128
    NUM_LABELS = 3

    try:
        tokenizer = BertTokenizer.from_pretrained(model_save_path)
        bert_model_core = TFBertModel.from_pretrained(MODEL_NAME)
        model = CustomBertForSequenceClassification(bert_model_core, num_labels=NUM_LABELS)

        dummy_inputs = {
            'input_ids': tf.zeros((1, MAX_LEN), dtype=tf.int32),
            'attention_mask': tf.zeros((1, MAX_LEN), dtype=tf.int32)
        }
        _ = model(dummy_inputs)

        model.load_weights(os.path.join(model_save_path, 'tf_model.h5'))
        st.success("감성 분석 모델과 토크나이저를 성공적으로 로드했습니다.")
        return tokenizer, model
    except Exception as e:
        st.error(f"감성 분석 모델 로드 중 오류 발생: {e}")
        st.info("모델 파일 경로를 확인해주세요.")
        st.stop()

model_save_directory = './my_custom_bert_sentiment_model'
if not os.path.exists(model_save_directory):
    st.error(f"모델 저장 경로 '{model_save_directory}'를 찾을 수 없습니다.")
    st.stop()

tokenizer, sentiment_model = load_sentiment_model(model_save_directory)

# --- 감성 예측 함수 ---
MAX_LEN = 128
label_mapping = {'부정': 0, '중립': 1, '긍정': 2}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def predict_sentiment_tf(text, model, tokenizer, max_len, reverse_label_mapping):
    encoded_input = tokenizer.encode_plus(
        str(text),
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='tf'
    )
    output_logits = model(encoded_input, training=False)
    probabilities = tf.nn.softmax(output_logits, axis=-1).numpy()[0]
    predicted_label_idx = np.argmax(probabilities)
    predicted_sentiment = reverse_label_mapping[predicted_label_idx]
    return predicted_sentiment, probabilities

# --- 세션 상태에서 선택된 도서 정보 가져오기 ---
if 'selected_book' not in st.session_state or st.session_state.selected_book is None:
    st.warning("분석할 도서를 선택해주세요. '도서 검색' 페이지로 돌아가세요.")
    st.page_link("main_app.py", label="도서 검색 페이지로 돌아가기", icon="🏠")
    st.stop()

selected_book = st.session_state.selected_book
st.subheader(f"'{selected_book['title']}' 의 리뷰 분석")
st.write(f"상세 URL: {selected_book['detail_url']}")

# --- 리뷰 크롤링 및 분석 기능 ---
if st.button("리뷰 크롤링 및 감성 분석 시작"):
    all_reviews_data = []
    current_page_num = 1

    with st.spinner("리뷰 데이터를 가져오고 감성을 분석 중입니다..."):
        try:
            driver.get(selected_book['detail_url'])
            review_list_selector = "#ReviewList1 > div.tab_wrap.type_sm > div.tab_content > div > div.comment_list"
            
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, review_list_selector))
            )
            
            while True:
                st.info(f"페이지 {current_page_num} 리뷰 크롤링 중...")

                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, f"{review_list_selector} > div.comment_item"))
                    )
                except TimeoutException:
                    if current_page_num == 1:
                        st.warning("이 도서에는 리뷰가 없습니다.")
                    else:
                        st.info(f"페이지 {current_page_num}에서 리뷰를 더 이상 찾을 수 없습니다.")
                    break

                try:
                    first_review_element_on_page = driver.find_element(By.CSS_SELECTOR, f"{review_list_selector} > div.comment_item")
                except NoSuchElementException:
                    st.info(f"페이지 {current_page_num}에서 리뷰 요소를 찾을 수 없어 크롤링을 종료합니다.")
                    break

                soup = BeautifulSoup(driver.page_source, 'html.parser')
                review_items_soup = soup.select(f"{review_list_selector} > div.comment_item")

                if not review_items_soup:
                    st.info(f"페이지 {current_page_num}에서 파싱된 리뷰가 없습니다. 크롤링을 종료합니다.")
                    break

                # 리뷰 데이터 추출 및 분석
                for item in review_items_soup:
                    # --- Start of Edit: 날짜 정보 수집 제거 ---
                    user_nickname_tag = item.select_one("div.user_info_box span.info_item")
                    user_nickname = user_nickname_tag.text.strip() if user_nickname_tag else "N/A"

                    rating_tag = item.select_one("span.caption-badge.caption-secondary")
                    rating = rating_tag.text.strip() if rating_tag else "N/A"

                    quote_tag = item.select_one("span.review_quotes_text")
                    quote = quote_tag.text.strip() if quote_tag else ""
                    
                    review_text_div = item.select_one("div.comment_text")
                    review_content = review_text_div.get_text(separator=" ", strip=True) if review_text_div else "내용 없음"

                    sentiment, probabilities = predict_sentiment_tf(
                        review_content, sentiment_model, tokenizer, MAX_LEN, reverse_label_mapping
                    )
                    all_reviews_data.append({
                        "닉네임": user_nickname, "평점": rating, "요약": quote,
                        "리뷰 내용": review_content, "예측 감성": sentiment,
                        "긍정 확률": f"{probabilities[2]:.4f}", "중립 확률": f"{probabilities[1]:.4f}", "부정 확률": f"{probabilities[0]:.4f}"
                    })
                    # --- End of Edit ---

                try:
                    next_button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn_page.next:not([disabled])"))
                    )
                    driver.execute_script("arguments[0].click();", next_button)
                    WebDriverWait(driver, 15).until(
                        EC.staleness_of(first_review_element_on_page)
                    )
                    current_page_num += 1
                    time.sleep(0.5)

                except (TimeoutException, NoSuchElementException):
                    st.info(f"마지막 페이지({current_page_num})에 도달했습니다. 총 {len(all_reviews_data)}개의 리뷰를 수집했습니다.")
                    break
           
            if all_reviews_data:
                df_reviews = pd.DataFrame(all_reviews_data)
                st.session_state.df_reviews = df_reviews
                st.success(f"총 {len(all_reviews_data)}개의 리뷰를 성공적으로 가져오고 분석했습니다.")
                st.dataframe(df_reviews, use_container_width=True)

                st.write("---")
                st.subheader("리뷰 감성 분포")
                fig_sentiment, ax_sentiment = plt.subplots(figsize=(8, 5))
                sentiment_counts = df_reviews['예측 감성'].value_counts().reindex(['긍정', '중립', '부정'], fill_value=0)
                sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax_sentiment, palette="viridis")
                ax_sentiment.set_title(f"'{selected_book['title']}' 리뷰 감성 분포", fontsize=16)
                for p in ax_sentiment.patches:
                    ax_sentiment.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
                st.pyplot(fig_sentiment)
                
                st.write("---")
                st.subheader("리뷰 워드클라우드")
                all_review_text = " ".join(df_reviews['리뷰 내용'].dropna().tolist())

                if valid_font_path:
                    if all_review_text:
                        filtered_words = [word for word in all_review_text.split() if len(word) > 1 and all('가' <= char <= '힣' for char in word)]
                        wc = WordCloud(
                            font_path=valid_font_path,
                            background_color="white",
                            width=800,
                            height=400,
                            max_words=100,
                            collocations=False
                        ).generate(" ".join(filtered_words))
                        
                        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis('off')
                        st.pyplot(fig_wc)
                    else:
                        st.info("워드클라우드를 생성할 리뷰 내용이 없습니다.")
                else:
                    st.warning("워드클라우드를 생성하기 위한 한글 폰트를 찾을 수 없습니다. 'fonts/NanumGothic.ttf' 파일을 추가하거나 시스템에 한글 폰트를 설치해주세요.")
                
                st.write("---")
                st.subheader("평점 분포")
                ratings_numeric = []
                for r in df_reviews['평점']:
                    try:
                        if '점 중' in r:
                            score_str = r.split('점 중')[1].strip().replace('점', '')
                            ratings_numeric.append(float(score_str))
                        elif r != "N/A":
                            ratings_numeric.append(float(r))
                    except ValueError:
                        continue
                if ratings_numeric:
                    fig_rating, ax_rating = plt.subplots(figsize=(8, 5))
                    sns.histplot(ratings_numeric, bins=10, kde=True, ax=ax_rating)
                    ax_rating.set_title(f"'{selected_book['title']}' 리뷰 평점 분포", fontsize=16)
                    ax_rating.set_xlabel("평점 (10점 만점)", fontsize=12)
                    ax_rating.set_ylabel("리뷰 개수", fontsize=12)
                    st.pyplot(fig_rating)
                else:
                    st.info("평점 데이터를 시각화할 수 없습니다.")

                st.write("---")
                st.subheader("감성별 샘플 리뷰")
                for sentiment_label in ['긍정', '중립', '부정']:
                    st.markdown(f"#### {sentiment_label} 리뷰")
                    sample_reviews = df_reviews[df_reviews['예측 감성'] == sentiment_label].head(3)
                    if not sample_reviews.empty:
                        for idx, row in sample_reviews.iterrows():
                            # --- Start of Edit: 샘플 리뷰 출력에서 날짜 제거 ---
                            st.markdown(f"**닉네임:** {row['닉네임']} | **평점:** {row['평점']} | **확률:** 긍정 {row['긍정 확률']}, 중립 {row['중립 확률']}, 부정 {row['부정 확률']}")
                            # --- End of Edit ---
                            if row['요약']:
                                st.markdown(f"**요약:** _{row['요약']}_")
                            st.write(f"**리뷰:** {row['리뷰 내용']}")
                            st.write("---")
                    else:
                        st.info(f"해당 감성의 리뷰가 없습니다.")

            else:
                st.warning("가져올 수 있는 리뷰가 없습니다.")

        except Exception as e:
            st.error(f"리뷰 크롤링 및 분석 중 오류가 발생했습니다: {e}")
            st.info("웹사이트 구조 변경, 네트워크 문제 등을 확인해주세요.")

# --- CSV 다운로드 기능 ---
if 'df_reviews' in st.session_state and not st.session_state.df_reviews.empty:
    csv_data = st.session_state.df_reviews.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="리뷰 데이터 CSV 파일로 다운로드",
        data=csv_data,
        file_name=f"{selected_book['title']}_reviews.csv",
        mime="text/csv",
    )

st.write("---")
st.page_link("main_app.py", label="새 도서 검색", icon="🏠")
