import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import time
import re

# 1. 검색어 입력
book_name = input("책 검색: ")
encoded_book_name = quote(book_name)
search_url = f"https://www.yes24.com/Product/Search?domain=BOOK&query={encoded_book_name}"

# 2. 검색 결과 페이지 요청
headers = {
    "User-Agent": "Mozilla/5.0"
}
response = requests.get(search_url, headers=headers)

if response.status_code == 200:
    search_soup = BeautifulSoup(response.text, 'html.parser')

    # 3. 첫 번째 검색 결과 링크
    first_link_tag = search_soup.select_one("ul#yesSchList li a.gd_name")
    if first_link_tag:
        product_url = "https://www.yes24.com" + first_link_tag['href']
        print(f"첫 번째 검색 결과 URL: {product_url}")
    else:
        print("검색 결과가 없습니다.")
        exit()

    # 4. 상품 상세 페이지 요청
    product_response = requests.get(product_url, headers=headers)
    if product_response.status_code == 200:
        product_soup = BeautifulSoup(product_response.text, 'html.parser')

        # 5. 책 제목 및 평점
        try:
            title = product_soup.select_one('h2.gd_name').text.strip()
        except:
            title = "(제목 없음)"
        try:
            score = product_soup.select_one('em.yes_b').text.strip()
        except:
            score = "(평점 없음)"

        # 6. 상품 ID 추출
        try:
            goods_id = product_url.split('/')[-1].split('?')[0]
        except:
            print("상품 ID 추출 실패")
            exit()

        # 7. AJAX 요청으로 리뷰 + 평점 수집
        print("\n책 제목:", title)
        print("평점:", score)
        print("한줄평:")

        page = 1
        all_reviews = []

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
                rating_tag = block.select_one('span.rating')
                
                if review_tag and rating_tag:
                    review_text = review_tag.text.strip()
                    
                    # 평점 숫자 추출 (예: "rating rating_5" → 5)
                    rating_class = rating_tag.get("class", [])
                    rating_score = None
                    for cls in rating_class:
                        if cls.startswith("rating_"):
                            rating_score = int(cls.split("_")[1])
                            break

                    all_reviews.append((review_text, rating_score))

            page += 1
            time.sleep(0.3)

        # 출력
        for i, (review, rating) in enumerate(all_reviews, 1):
            print(f"{i}. ({rating}점) {review}")

    else:
        print("상품 상세 페이지 요청 실패")
else:
    print("검색 결과 페이지 요청 실패")
