


import os

# 새로운 메모장을 열어 특정 파일을 생성
file_path = f"C:\\Users\\{os.getlogin()}\\Desktop\\진실의 눈.txt"
with open(file_path, "w", encoding="utf-8") as f:
    f.write("This program is for news judgement. \n\nPlease follow the instructions. \nThe results will show as (Real/Fake) and Percentage %")

# 메모장을 열어서 해당 파일 표시
os.system(f"notepad {file_path}")




"""
진실의 눈 =========================================================

1. Url 입력받기
2. Chatgpt API로 키워드 추출
3. 네이버 뉴스 API로 유사사 뉴스 추출
4. KLUE RoBERTa fine-tuned 모델로 뉴스 진위 판단 (아직 fine-tuning 필요)
5. 결과 출력
"""



#==========================================================================================================



import os
import ctypes
if ctypes.windll.user32.MessageBoxW(0, "Please wait for modules to load...", "Program", 0x30 | 0x1) == 0x1:
    pass
else:
    exit()

# 설치할 패키지 목록
packages = [
    "youtube-transcript-api",
    "openai",
    "transformers",
    "beautifulsoup4",
    "transformers",
    "datasets",
    "scikit-learn",
    "pandas",
    "torch"
]

# 패키지 설치
for package in packages:
    os.system(f"pip install {package}")


from youtube_transcript_api import YouTubeTranscriptApi
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import openai
import requests
from bs4 import BeautifulSoup
import webbrowser
print("\n" * 10)


client_id = '네이버_API_Client_ID'
client_secret = '네이버_API_Client_Secret'
openai.api_key = 'Open ai 키'
#==========================================================================================================


def question_chatgpt(question):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        max_tokens=120,
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )
    return response['choices'][0]['message']['content'].strip()











#==========================================================================================================



class Pick_news:
    def __init__(self):
        self.url = None
        self.news = None

    def get_naver_url(self):
        global news, url_find
        def crawl_naver_news(url):
            headers = {
                "User-Agent": "Mozilla/5.0"  
            }

            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            # 본문
            content_tag = soup.find('article', {'id': 'dic_area'})
            content = content_tag.get_text().strip() if content_tag else ctypes.windll.user32.MessageBoxW(0, "No content found", "Error", 0x10 | 0x1)

            if not content_tag:
                exit()

            return content

        url = input("Naver news URL: ")
        news = crawl_naver_news(url)
        url_find = url


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
        
    def get_news(self):
        global news
        news = input("Please enter the news manually: ")





Pick_news = Pick_news()






#===========================================================================================================




if ctypes.windll.user32.MessageBoxW(0, 'To start "진실의 눈" press (Y)', "Program", 0x20 | 0x4) == 0x6:
    pass
else:
    exit()

articleTemp = ctypes.windll.user32.MessageBoxW(0, 'Article URL [Naver]: (Y), YouTube: (N), Text: (Cancel button)', "Program", 0x20 | 0x3)

if articleTemp == 0x6:  
    Pick_news.get_naver_url()



elif articleTemp == 0x7:
    Pick_news.get_youtube_url()
    

elif articleTemp == 0x2:
    Pick_news.get_news()


else:
    ctypes.windll.user32.MessageBoxW(0, "Invalid option", "Error", 0x10 | 0x1)
    exit()

#==========================================================================================================



chatgpt_keywords = question_chatgpt(news + "\n\n이 뉴스를 키워드 5개 이하로 요약. 다른 말은 하지마. ex) 000, 000, 000, 000, 000")



#==========================================================================================================




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



#==========================================================================================================



chatgpt_percentage = int(question_chatgpt(f"{news}\n이 문장과 \n{another_news}\n이 문장의 유사도를 평가해줘. 다른 말은 하지말고 백분율 숫자만 말해줘."))


#===========================================================================================================
# Needs to be changed. (fine-tuning) 

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import pandas as pd

def load_and_prepare_data(file_path, text_col, label_col):
    df = pd.read_csv(file_path)
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].astype(int)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"], df["label"], test_size=0.1, random_state=42)
    train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
    test_dataset = Dataset.from_dict({"text": test_texts.tolist(), "label": test_labels.tolist()})
    return train_dataset, test_dataset

def preprocess(tokenizer, dataset):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)
    tokenized = dataset.map(preprocess_function, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

def train_model(train_dataset, test_dataset, model_name, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_train = preprocess(tokenizer, train_dataset)
    tokenized_test = preprocess(tokenizer, test_dataset)

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
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    trainer.train()
    trainer.save_model(output_dir)

model_name = "klue/roberta-base"

# 1) NSMC
train_nsmc, test_nsmc = load_and_prepare_data("nsmc.csv", "document", "label")
train_model(train_nsmc, test_nsmc, model_name, "./finetuned_klue_nsmc")

# 2) HateSpeech
train_hate, test_hate = load_and_prepare_data("hatespeech.csv", "sentence", "label")
train_model(train_hate, test_hate, model_name, "./finetuned_klue_hatespeech")

# 3) Clickbait
train_clickbait, test_clickbait = load_and_prepare_data("clickbait.csv", "title", "label")
train_model(train_clickbait, test_clickbait, model_name, "./finetuned_klue_clickbait")
