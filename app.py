import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

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
      bot_msg = "대답"

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