import gradio as gr

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