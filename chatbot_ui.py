import gradio as gr
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

def respond(message, history):
    # Add user's message to history
    history = history or []
    inputs = tokenizer([message], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    
    # Append to history
    history.append(("You: " + message, "Sowmya: " + response))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("<h1>💬 Chat with Sowmya</h1>")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message", placeholder="Type a message and hit Enter")
    state = gr.State([])  # Stores chat history
    
    def on_submit(user_message, chat_history):
        return respond(user_message, chat_history)
    
    msg.submit(on_submit, inputs=[msg, state], outputs=[chatbot, state])

demo.launch()
#demo.launch(share=True)


