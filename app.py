import os
os.system("pip install -U langchain langchain-community faiss-cpu sentence-transformers --no-cache-dir")

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import gradio as gr

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local(".", embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 1})


from huggingface_hub import InferenceClient
client = InferenceClient(
    "HuggingFaceH4/zephyr-7b-beta",
    token=os.environ.get("hftoken")  
)
def get_context(query):
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)[:800]  


def build_prompt(context, user_input):
    return f"""You are a warm, understanding, and insightful mental health assistant.
The user is experiencing an emotional or psychological difficulty. Your job is to help them understand and cope with their situation in a supportive, non-judgmental, and structured way.
Please respond to their concern using the following 5-step structure. Each step should build upon the previous, offering reflection, clarity, and actionable support:
Step 1: **Acknowledge the Feeling**  
Begin by gently acknowledging and validating the user's emotional experience. Show empathy and help them feel heard and understood.
Step 2: **Explore Possible Causes**  
Offer insight into what might be causing or contributing to the issue, based on general psychological principles or relatable human patterns.
Step 3: **Normalize the Experience**  
Reassure the user that they are not alone in feeling this way, and that such responses are common among people in similar situations.
Step 4: **Suggest Practical Coping Strategies**  
Give 2–3 realistic, actionable strategies the user can try to better manage or ease their emotional state.
Step 5: **Reframe the Perspective**  
Help the user look at their situation from a new, empowering point of view. Leave them with a hopeful or encouraging thought.
Be supportive, clear, and use accessible language. The tone should feel like a kind, emotionally intelligent therapist helping a friend.
Look at this example below for reference.
---
Example:
User's Concern: I constantly feel like I’m not good enough, even when I succeed.
Step 1: 
It’s completely understandable to feel this way. That constant self-doubt can be exhausting, and it's okay to admit that you're struggling with these emotions.
Step 2:   
This feeling might stem from perfectionism, childhood criticism, or comparing yourself to others. Sometimes, even after success, we move the goalpost instead of recognizing our progress.
Step 3:   
Many high-achieving and thoughtful people experience this. You're not alone — self-doubt is more common than most people realize, especially in a world that constantly pushes us to “do more.”
Step 4:  
Try writing down a few accomplishments at the end of each week. Speak to yourself the way you would comfort a friend — with kindness, not criticism. You might also explore therapy or coaching to uncover deeper patterns and reframe your self-image.
Step 5:   
The fact that you're reflecting on this shows a high level of self-awareness. Progress is not about proving you're enough — it’s about realizing that you already are.
    *You can give a little longer answer where you feel like its necessary.
---
Now, based on the new user's concern below, provide your own thoughtful, 5-step response. And end with a brief summary.
User's Concern: {user_input}
    Context:
{context}
User's Concern: {user_input}
Step 1:
Step 2:
Step 3:
Step 4:
Step 5:
"""

from gtts import gTTS
import uuid
import os

def respond(message, history):
    system_message = "You are a kind and emotionally intelligent mental health companion."
    max_tokens = 512
    temperature = 0.7
    top_p = 0.95


    context = get_context(message)
    prompt = build_prompt(context, message)

    messages = [{"role": "system", "content": system_message}]
    for turn in history:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            user, assistant = turn
            if user:
                messages.append({"role": "user", "content": user})
            if assistant:
                messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": prompt})

    response_text = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    ):
        token = message.choices[0].delta.content or ""
        response_text += token

   
    tts = gTTS(text=response_text)
    audio_path = f"response.mp3"
    tts.save(audio_path)

    return audio_path, history  


  

demo = gr.Interface(
    fn=respond,
    inputs=[
        gr.Textbox(label="Your message"),
        gr.State([]),
    ],
    outputs=[
        gr.Audio(type="filepath", label="Therapy Response (Audio)"),
        gr.State(),
    ],
    title="🧠 Therapy Companion (Voice)",
    description="Speak with an emotionally intelligent LLM. Responses are now read out to you!"
)

if __name__ == "__main__":
    demo.launch()