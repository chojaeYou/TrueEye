import os

# 새로운 메모장을 열어 특정 파일을 생성
file_path = f"C:\\Users\\{os.getlogin()}\\Desktop\\memo.txt"
with open(file_path, "w", encoding="utf-8") as f:
    f.write("This program is for news judgement. \n\nPlease follow the instructions. \nThe results will show as (Real/Fake) percentage %")

# 메모장을 열어서 해당 파일 표시
os.system(f"notepad {file_path}")




"""
진실의 눈 =========================================================

1. Url 입력받기
2. Claude API로 뉴스 요약 및 키워드 추출
3. KLUE RoBERTa 모델로 뉴스 진위 판단
4. 결과 출력
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
    "anthropic",
    "transformers",
    "beautifulsoup4"
]

# 패키지 설치
for package in packages:
    os.system(f"pip install {package}")


from youtube_transcript_api import YouTubeTranscriptApi
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import anthropic
import requests
from bs4 import BeautifulSoup
#===========================================================================================================


print("\n" * 10)

if ctypes.windll.user32.MessageBoxW(0, 'To start "진실의 눈" press (Y)', "Program", 0x20 | 0x4) == 0x6:
    pass
else:
    exit()

articleTemp = ctypes.windll.user32.MessageBoxW(0, 'Article URL [Naver]: (Y), YouTube: (N), Text: (C)', "Program", 0x20 | 0x4)

if articleTemp == 0x6:  
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







elif articleTemp == 0x7:

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

elif articleTemp == 0x2:
    news = input("Please enter the news manually: ")



else:
    ctypes.windll.user32.MessageBoxW(0, "Invalid option", "Error", 0x10 | 0x1)
    exit()

#==========================================================================================================

# Claude API 키 설정
client = anthropic.Anthropic(
    api_key="your-claude-api-key"
)

# 뉴스 기사 전문
news = "여기에 뉴스 전문 텍스트를 넣어줘"

# Claude 3 Opus를 이용해 뉴스 요약 요청
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=120,
    temperature=0.7,
    messages=[
        {
            "role": "user",
            "content": news + "\n\n이 뉴스를 '구체적이고 정확한 한줄'로 (약 400자) 요약한 문장만 보여줘. 다른 말은 하지마. 요약 문장만. (뉴스 내용만)"
        }
    ]
)

# 결과 출력
claude_summation = response.content[0].text.strip()


#==========================================================================================================


# KLUE RoBERTa 분류 모델 불러오기
model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-base", num_labels=2)

# tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

# 파이프라인 생성
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


# 분류 실행
result = classifier(claude_summation)[0]

# 레이블 맵핑 (0 = 진짜 뉴스, 1 = 가짜 뉴스)
label_map = {
    "LABEL_0": "Real",
    "LABEL_1": "Fake"
}

# 결과 출력
print(f"\n Result: {label_map[result['label']]} ({result['score']:.2%})")
