import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
# -------------------------
# 모델 및 토크나이저 로드
# -------------------------
@st.cache_resource
def load_sentiment_model():
    model = load_model('C:/Lec/_DeepNLP25/Crawler/Yes24Crawler/model/best_model_lstm_okt.h5')
    tokenizer = joblib.load('C:/Lec/_DeepNLP25/Crawler/Yes24Crawler/model/yes24_tokenizer.pkl')
    return model, tokenizer

model, tokenizer = load_sentiment_model()
max_len = 50
label_map_reverse = {0: 'negative', 1: 'neutral', 2: 'positive'}

# -------------------------
# 리뷰 크롤링 함수
# -------------------------
def crawl_yes24_reviews(book_name):
    headers = {"User-Agent": "Mozilla/5.0"}
    encoded_book_name = quote(book_name)
    search_url = f"https://www.yes24.com/Product/Search?domain=BOOK&query={encoded_book_name}"

    res = requests.get(search_url, headers=headers)
    if res.status_code != 200:
        return []

    soup = BeautifulSoup(res.text, 'html.parser')
    first_link = soup.select_one("ul#yesSchList li a.gd_name")
    if not first_link:
        return []

    product_url = "https://www.yes24.com" + first_link['href']
    goods_id = product_url.split('/')[-1].split('?')[0]

    reviews = []
    page = 1

    while True:
        timestamp = int(time.time() * 1000)
        ajax_url = f"https://www.yes24.com/Product/communityModules/AwordReviewList/{goods_id}?goodsSetYn=N&DojungAfterBuy=0&Sort=2&_={timestamp}&PageNumber={page}"
        ajax_res = requests.get(ajax_url, headers=headers)
        if ajax_res.status_code != 200 or not ajax_res.text.strip():
            break

        ajax_soup = BeautifulSoup(ajax_res.text, 'html.parser')
        review_blocks = ajax_soup.select('div.cmtInfoGrp')
        if not review_blocks:
            break

        for block in review_blocks:
            review_tag = block.select_one('span.txt')
            if review_tag:
                review_text = review_tag.text.strip()
                if review_text:
                    reviews.append(review_text)
        page += 1
        time.sleep(0.2)
    
    return reviews

# -------------------------
# 감성 분석 함수
# -------------------------
def predict_sentiment(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    seqs = [seq for seq in seqs if len(seq) > 0]  # 빈 시퀀스 제거
    padded = pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')
    preds = model.predict(padded)
    labels = np.argmax(preds, axis=1)
    return labels

# -------------------------
# Streamlit 앱 시작
# -------------------------
st.title("YES24 책 리뷰 감성 분석 대시보드")

book_name = st.text_input("책 제목을 입력하세요", value="미드나잇 라이브러리")

if st.button("분석 시작"):
    with st.spinner("리뷰 크롤링 중..."):
        reviews = crawl_yes24_reviews(book_name)

    if not reviews:
        st.error("리뷰를 가져올 수 없습니다. 책 제목을 다시 확인해주세요.")
    else:
        st.success(f"리뷰 {len(reviews)}개 수집 완료!")

        with st.spinner("감성 분석 중..."):
            predicted_labels = predict_sentiment(reviews)

        # 라벨 매핑 및 데이터프레임 생성
        label_names = [label_map_reverse[label] for label in predicted_labels]
        df_result = pd.DataFrame({
            'review': reviews[:len(label_names)],
            'sentiment': label_names
        })

        # 시각화
        st.subheader("감성 분석 결과 분포")
        sentiment_counts = df_result['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

        # 퍼센트 출력
        st.subheader("감성 비율 (%)")
        total = len(label_names)
        for label in ['positive', 'neutral', 'negative']:
            percent = (sentiment_counts.get(label, 0) / total) * 100
            st.write(f"**{label.capitalize()}**: {percent:.2f}%")

        # 표로 리뷰 출력
        st.subheader("리뷰 및 예측 결과")
        st.dataframe(df_result)
