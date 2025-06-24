import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.font_manager as fm

# --- 한글 폰트 설정을 위한 임포트 및 함수 (Analyze_Reviews.py와 동일) ---

# 폰트 파일 경로 설정
FONT_PATH = "fonts/NanumGothic.ttf"

@st.cache_resource
def get_font_path():
    """
    사용 가능한 한글 폰트 경로를 반환합니다.
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
    page_title="CSV 분석",
    page_icon="📄",
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

# --- 메인 페이지 ---
st.title("📄 CSV 파일 시각화")
st.info("이전에 분석하고 다운로드한 리뷰 CSV 파일을 업로드하여 시각화 결과를 다시 확인하세요.")

uploaded_file = st.file_uploader("CSV 파일 선택", type=["csv"])

if uploaded_file is not None:
    try:
        df_reviews = pd.read_csv(uploaded_file)
        st.success("CSV 파일을 성공적으로 로드했습니다.")
        st.dataframe(df_reviews, use_container_width=True)

        # 필요한 컬럼이 있는지 확인
        required_columns = ['예측 감성', '리뷰 내용', '평점']
        if not all(col in df_reviews.columns for col in required_columns):
            st.error(f"업로드한 CSV 파일에 필요한 컬럼({', '.join(required_columns)})이 모두 존재하지 않습니다. 올바른 파일을 업로드했는지 확인해주세요.")
        else:
            book_title = os.path.splitext(uploaded_file.name)[0].replace('_reviews', '')

            # --- 시각화 섹션 ---
            st.write("---")
            st.subheader("리뷰 감성 분포")
            fig_sentiment, ax_sentiment = plt.subplots(figsize=(8, 5))
            sentiment_counts = df_reviews['예측 감성'].value_counts().reindex(['긍정', '중립', '부정'], fill_value=0)
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax_sentiment, palette="viridis")
            ax_sentiment.set_title(f"'{book_title}' 리뷰 감성 분포", fontsize=16)
            for p in ax_sentiment.patches:
                ax_sentiment.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
            st.pyplot(fig_sentiment)
            
            st.write("---")
            st.subheader("리뷰 워드클라우드")
            all_review_text = " ".join(df_reviews['리뷰 내용'].dropna().tolist())

            if valid_font_path:
                if all_review_text:
                    filtered_words = [word for word in all_review_text.split() if len(word) > 1 and all('가' <= char <= '힣' for char in word)]
                    if filtered_words:
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
                        st.info("워드클라우드를 생성할 단어가 없습니다.")
                else:
                    st.info("워드클라우드를 생성할 리뷰 내용이 없습니다.")
            else:
                st.warning("워드클라우드를 생성하기 위한 한글 폰트를 찾을 수 없습니다. 'fonts/NanumGothic.ttf' 파일을 추가하거나 시스템에 한글 폰트를 설치해주세요.")
            
            st.write("---")
            st.subheader("평점 분포")
            ratings_numeric = []
            for r in df_reviews['평점'].dropna():
                try:
                    if '점 중' in str(r):
                        score_str = str(r).split('점 중')[1].strip().replace('점', '')
                        ratings_numeric.append(float(score_str))
                    elif str(r) != "N/A":
                        ratings_numeric.append(float(r))
                except (ValueError, IndexError):
                    continue
            
            if ratings_numeric:
                fig_rating, ax_rating = plt.subplots(figsize=(8, 5))
                sns.histplot(ratings_numeric, bins=10, kde=True, ax=ax_rating)
                ax_rating.set_title(f"'{book_title}' 리뷰 평점 분포", fontsize=16)
                ax_rating.set_xlabel("평점 (10점 만점)", fontsize=12)
                ax_rating.set_ylabel("리뷰 개수", fontsize=12)
                st.pyplot(fig_rating)
            else:
                st.info("평점 데이터를 시각화할 수 없습니다.")

    except Exception as e:
        st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")

st.write("---")
st.page_link("main_app.py", label="새 도서 검색하러 가기", icon="🏠")
