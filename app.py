import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel, PeftConfig

import requests
from bs4 import BeautifulSoup

import os
from dotenv import load_dotenv
load_dotenv()

### LLM 모델 불러오기
peft_model_id = "ChaeMs/KoRani-5.8b"

config = PeftConfig.from_pretrained(peft_model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                             quantization_config=bnb_config,
                                             device_map="auto")

model = PeftModel.from_pretrained(model, peft_model_id)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model.eval()
########################################################################



### Functions
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

def build_prompt(user_input):
    return f"### User\n{user_input}\n답변은 반드시 완성된 문장으로 생성해줘.\n\n### Bot\n"


def make_answer(text):
    result = pipe(build_prompt(text),
                   return_full_text=False,
                   do_sample=False,
                   repetition_penalty=1.2,
                   temperature=0.1,
                   max_new_tokens=256)
    return result[0]['generated_text'].split("###")[0]

########################################################################



### Google Search
def scrap_google_news(keyword: str, limit=10):
    google_search_url = 'https://www.google.com/search'

    params = {'q': keyword, 'tbm': 'nws', 'num': limit}

    headers = {"User-Agent": os.getenv("USER_AGENT_INFO")}

    res = requests.get(google_search_url, params=params, headers=headers)
    soup = BeautifulSoup(res.content, 'html.parser')

    news_results = []
    for el in soup.select("div.SoaBEf"):
        news_results.append(
            {
                "link": el.find("a")["href"],
                "title": el.select_one("div.MBeuO").get_text(),
                "date": el.select_one(".OSrXXb").get_text()
            }
        )

    back_ground_info = ''
    refs = []
    for i in news_results:
        back_ground_info += (i['title'] + '\n')
        refs.append((i['date'],i['link']))

    return back_ground_info, refs


### gradio

def user(message, history):
    return "", history + [[message, None]]


def bot(history):
    if history[-1][0][0] == "/":
        bot_msg = "검색"
        user_msg = history[-1][0][1:]
        history[-1] = [user_msg, bot_msg]
        return history
    else:
      bot_msg = make_answer(history[-1][0])
      user_msg = history[-1][0]
      history[-1] = [user_msg, bot_msg]
      return history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder='검색을 하려면 "/"를 먼저 입력, (예시: "/세계 경제가 어때?")')
    clear = gr.Button("Clear")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()