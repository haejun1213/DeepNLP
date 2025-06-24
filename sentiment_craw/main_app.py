import streamlit as st
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse
import time
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="도서 검색",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 도서 검색")
st.write("분석하고 싶은 도서명을 입력하고 검색하세요.")

# 세션 상태 초기화
if 'selected_book' not in st.session_state:
    st.session_state.selected_book = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = pd.DataFrame()

# --- Selenium WebDriver 설정 (캐싱) ---
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

# --- 도서 검색 기능 ---
book_keyword = st.text_input("도서명 입력", placeholder="예: 데미안")

if st.button("도서 검색"):
    if not book_keyword:
        st.warning("도서명을 입력해주세요.")
        st.session_state.selected_book = None
        st.session_state.search_results = pd.DataFrame()
        st.stop()

    search_url = f"https://search.kyobobook.co.kr/search?keyword={book_keyword}"
    st.info(f"'{book_keyword}' 도서를 검색 중입니다...")
    
    try:
        driver.get(search_url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "ul.prod_list")) 
        )
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        book_items = soup.select("ul.prod_list > li.prod_item") 
        
        if not book_items:
            st.warning(f"'{book_keyword}'에 대한 검색 결과를 찾을 수 없습니다.")
            st.session_state.search_results = pd.DataFrame()
            st.session_state.selected_book = None
        else:
            search_results_list = []
            for item in book_items:
                link_tag = item.select_one("a.prod_info")
                if link_tag and link_tag.get('href'):
                    detail_url = link_tag.get('href')
                    
                    title_tag = link_tag.select_one("span[id^='cmdtName_']")
                    title = title_tag.text.strip() if title_tag else "정보 없음"
                    
                    # --- Start of Edit: 저자 및 출판사 정보 파싱 로직 수정 ---
                    # 저자 정보 가져오기 (여러 명일 경우 모두 포함)
                    author_tags = item.select("div.prod_author_info a.author")
                    author = ', '.join([tag.text.strip() for tag in author_tags]) if author_tags else "정보 없음"

                    # 출판사 및 발행일 정보 가져오기
                    publisher_tag = item.select_one("div.prod_publish a.text")
                    date_tag = item.select_one("div.prod_publish span.date")
                    
                    publisher_text = publisher_tag.text.strip() if publisher_tag else ""
                    date_text = date_tag.text.strip() if date_tag else ""
                    
                    if publisher_text and date_text:
                        publisher = f"{publisher_text} ({date_text})"
                    elif publisher_text:
                        publisher = publisher_text
                    else:
                        publisher = "정보 없음"
                    # --- End of Edit ---
                    
                    parsed_url = urlparse(detail_url)
                    product_id = parsed_url.path.split('/')[-1]
                    if not product_id: 
                        product_id = parsed_url.path.split('/')[-2]

                    search_results_list.append({
                        "도서명": title, "저자": author, "출판사": publisher,
                        "product_id": product_id, "상세 URL": detail_url
                    })
            
            if search_results_list:
                st.session_state.search_results = pd.DataFrame(search_results_list)
                st.session_state.selected_book = None
            else:
                st.warning(f"'{book_keyword}'에 대한 유효한 도서 정보를 찾을 수 없습니다.")
                st.session_state.search_results = pd.DataFrame()
                st.session_state.selected_book = None

    except Exception as e:
        st.error(f"도서 검색 중 오류가 발생했습니다: {e}")
        st.session_state.search_results = pd.DataFrame()
        st.session_state.selected_book = None

# --- 검색 결과 표시 및 도서 선택 ---
if not st.session_state.search_results.empty:
    st.subheader("검색 결과")
    st.dataframe(st.session_state.search_results, use_container_width=True, hide_index=True)

    st.write("---")
    st.subheader("리뷰를 분석할 도서 선택 (상위 5개)")
    top_5_results = st.session_state.search_results.head(5)

    radio_options = [f"{row['도서명']} (저자: {row['저자']})" for index, row in top_5_results.iterrows()]
    radio_values = list(top_5_results.index)

    default_radio_selection_index = 0
    
    selected_radio_index = st.radio(
        "도서를 선택하세요:",
        options=radio_values,
        format_func=lambda x: radio_options[radio_values.index(x)],
        index=default_radio_selection_index
    )

    if selected_radio_index is not None:
        selected_book_data = st.session_state.search_results.loc[selected_radio_index]
        st.session_state.selected_book = {
            "title": selected_book_data['도서명'],
            "product_id": selected_book_data['product_id'],
            "detail_url": selected_book_data['상세 URL']
        }
        st.success(f"'{st.session_state.selected_book['title']}' 이(가) 선택되었습니다.")
    
    st.write("---")

    if st.session_state.selected_book:
        # pages 폴더가 있다고 가정
        st.page_link("pages/Analyze_Reviews.py", label="선택 도서 리뷰 분석하기", icon="📊")
    else:
        st.info("리뷰를 분석할 도서를 위에서 선택해주세요.")
