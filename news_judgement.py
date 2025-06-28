import os
import ctypes

# 모듈 설치 함수 (os로)
def install_packages():
    packages = [
        "youtube-transcript-api",
        "openai",
        "transformers",
        "beautifulsoup4",
        "datasets",
        "scikit-learn",
        "pandas",
        "torch"
    ]
    
    for package in packages:
        os.system(f"pip install {package}")

# 첫 화면 메시지 박스
if ctypes.windll.user32.MessageBoxW(0, "Please wait for modules to load...", "Program", 0x30 | 0x1) == 0x1:
    install_packages()

# 이후 코드 (클래스와 함수들)
import requests
import re
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import openai
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

# 기본 세팅: API 키 설정
client_id = '네이버_API_Client_ID'
client_secret = '네이버_API_Client_Secret'
openai.api_key = 'Open ai 키'

# ChatGPT로 키워드 추출 함수
def question_chatgpt(question):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        max_tokens=120,
        temperature=0.7,
        messages=[{
            "role": "user",
            "content": question
        }]
    )
    return response['choices'][0]['message']['content'].strip()

# Pick_news 클래스 (뉴스를 URL 또는 수동 입력 받는 클래스)
class Pick_news:
    def __init__(self):
        self.url = None
        self.news = None

    # 네이버 뉴스 URL로 뉴스 추출
    def get_naver_url(self):
        global news, url_find
        def crawl_naver_news(url):
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            content_tag = soup.find('article', {'id': 'dic_area'})
            content = content_tag.get_text().strip() if content_tag else None
            if not content_tag:
                ctypes.windll.user32.MessageBoxW(0, "No content found", "Error", 0x10 | 0x1)
                exit()
            return content
        url = input("Naver news URL: ")
        news = crawl_naver_news(url)
        url_find = url

    # 유튜브 URL로 뉴스 추출
    def get_youtube_url(self):
        global news, url_find
        def extract_video_id(url):
            match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
            if not match:
                ctypes.windll.user32.MessageBoxW(0, "No URL found", "Error", 0x10 | 0x1)
                exit()
            return match.group(1)

        video_url = input("YouTube video URL: ")
        video_id = extract_video_id(video_url)

        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
            news = "\n".join([entry['text'] for entry in transcript])
        except Exception as e:
            ctypes.windll.user32.MessageBoxW(0, f"Fail: {str(e)}", "Error", 0x10 | 0x1)
            exit()
        url_find = video_url

    # 뉴스 직접 입력 받기
    def get_news(self):
        global news
        news = input("Please enter the news manually: ")

# 뉴스 클래스 객체 생성
pick_news = Pick_news()

# 첫 화면에서 어떤 뉴스 소스를 사용할지 묻는 메시지 박스
articleTemp = ctypes.windll.user32.MessageBoxW(0, 'Article URL [Naver]: (Y), YouTube: (N), Text: (Cancel button)', "Program", 0x20 | 0x3)

if articleTemp == 0x6:  # Naver 뉴스
    pick_news.get_naver_url()
elif articleTemp == 0x7:  # YouTube 뉴스
    pick_news.get_youtube_url()
elif articleTemp == 0x2:  # 수동 입력
    pick_news.get_news()
else:
    ctypes.windll.user32.MessageBoxW(0, "Invalid option", "Error", 0x10 | 0x1)
    exit()

# ChatGPT로 키워드 추출
chatgpt_keywords = question_chatgpt(news + "\n\n이 뉴스를 키워드 5개 이하로 요약. 다른 말은 하지마. ex) 000, 000, 000, 000, 000")

# 네이버 뉴스에서 키워드 관련 뉴스 가져오기
def get_representative_news(keywords_string, client_id, client_secret):
    headers = {
        'X-Naver-Client-Id': client_id,
        'X-Naver-Client-Secret': client_secret
    }

    keywords = [kw.strip() for kw in keywords_string.split(",") if kw.strip()]
    all_links = []

    for kw in keywords:
        url = f'https://openapi.naver.com/v1/search/news.json?query={kw}&display=5&sort=date'
        try:
            response = requests.get(url, headers=headers)
            items = response.json().get('items', [])
            all_links.extend([item['link'] for item in items])
        except:
            continue

    def crawl(url):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            tag = soup.find('article', {'id': 'dic_area'})
            return tag.get_text().strip() if tag else None
        except:
            return None

    contents = [crawl(link) for link in all_links]
    contents = [c for c in contents if c]

    if not contents:
        ctypes.windll.user32.MessageBoxW(0, "No content found", "Error", 0x10 | 0x1)
        return None

    scores = [sum(c.count(k) for k in keywords) for c in contents]
    best = contents[scores.index(max(scores))]
    return best

keywords = chatgpt_keywords
another_news = get_representative_news(keywords, client_id, client_secret)

# ChatGPT로 유사도 평가
chatgpt_percentage = int(question_chatgpt(f"{news}\n이 문장과 \n{another_news}\n이 문장의 유사도를 평가해줘. 다른 말은 하지말고 백분율 숫자만 말해줘. 예: 75"))

# ML 모델 학습 및 평가 (예시)
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def load_and_prepare_data(file_path, text_col, label_col):
    df = pd.read_csv(file_path)
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].astype(int)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"], df["label"], test_size=0.1, random_state=42)
    train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
    test_dataset = Dataset.from_dict({"text": test_texts.tolist(), "label": test_labels.tolist()})
    return train_dataset, test_dataset

# 모델 학습 및 예측 함수 (예시)
def train_model(train_dataset, test_dataset, model_name, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=100,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    return model, tokenizer

# 예측용 함수
def predict(text, model, tokenizer):
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    result = classifier(text)[0]
    return result['score']

# 예측 결과 출력
print(f"전체: {(chatgpt_percentage * 0.4) + ((100 - model_a) * 0.25) + ((100 - model_b) * 0.2) + ((100 - model_c) * 0.15)}")
