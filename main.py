from flask import Flask, request, render_template
from bs4 import BeautifulSoup
import requests
import re
from transformers import BertTokenizer

app = Flask(__name__)

# 한글 불용어 리스트 (추가해야할듯???)
STOPWORDS = ['은', '는', '이', '가', '에', '을', '를', '도', '으로', '의', '와', '과']

# KLUE BERT 
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/do_something', methods=['POST'])
def do_something():
    url = request.form['url']
    print(f"url : {url}")
    
    title, content = crawl_naver_news(url)

    if title == "제목 없음" and content == "본문 없음":
        return render_template("error.html")

    # 한글 전처리
    clean_title = preprocess_korean(title)
    clean_content = preprocess_korean(content)

    # BERT
    title_tokens = tokenize_text(clean_title)
    content_tokens = tokenize_text(clean_content)

    return render_template(
        "analyze.html",
        mesg=url,
        title=clean_title,
        content=clean_content,
        title_tokens=title_tokens,
        content_tokens=content_tokens
    )

def crawl_naver_news(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        title_tag = soup.find('h2', class_='media_end_head_headline')
        title = title_tag.get_text().strip() if title_tag else "제목 없음"

        content_tag = soup.find('article', id='dic_area')
        content = content_tag.get_text().strip() if content_tag else "본문 없음"

        return title, content
    except Exception as e:
        print(f"크롤링 오류: {e}")
        return "제목 없음", "본문 없음"

def preprocess_korean(text):
    text = re.sub(r"[^\uAC00-\uD7A3\u0030-\u0039\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return " ".join(words)

def tokenize_text(text):
    return tokenizer.tokenize(text)

if __name__ == '__main__':
    app.run(debug=True)

