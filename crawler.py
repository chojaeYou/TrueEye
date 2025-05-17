import requests
from bs4 import BeautifulSoup

def crawl_naver_news(url):
    headers = {
        "User-Agent": "Mozilla/5.0"  #먼지는모르겠음
    }

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 제목
    title_tag = soup.find('h2', {'class': 'media_end_head_headline'})
    title = title_tag.get_text().strip() if title_tag else "제목 없음"

    # 본문
    content_tag = soup.find('article', {'id': 'dic_area'})
    content = content_tag.get_text().strip() if content_tag else "본문 없음"

    return title, content

url = "https://n.news.naver.com/mnews/article/001/0014608627"
title, content = crawl_naver_news(url)
print("제목:", title)
print("본문:", content)