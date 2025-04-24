# 🧠 Therapy Companion — Voice-Based AI Support

**Therapy Companion** is an AI-powered voice-based mental health assistant designed to provide empathetic, structured guidance through emotional challenges. It listens to user concerns, generates a supportive 5-step response, and reads the answer aloud using text-to-speech (gTTS).

---

## 🔥 Key Features

- 💬 Powered by **Zephyr-7B** large language model (via Hugging Face)
- 🎙️ Speaks responses using **gTTS** (Google Text-to-Speech)
- 📚 Knowledge base stored in FAISS vector database
- 🧠 5-step therapy prompt (acknowledge → explore → normalize → suggest → reframe)
- 🚀 Deployed live via **Gradio** on Hugging Face Spaces

---

## 🛠 Tech Stack

- `LangChain` for document retrieval & agent logic
- `FAISS` for semantic search in knowledge base
- `gTTS` for speech synthesis
- `Gradio` for frontend
- `HuggingFaceH4/zephyr-7b-beta` model via `InferenceClient`
- `Python`, `Colab`, `Hugging Face Spaces`

---

## 🚀 Live Demo

🧪 Try it now:  
👉 (https://huggingface.co/spaces/sarmadshakil666/Therapy_companion1)

---

graph TD
    A[User Query] -->|Text Input| B[Gradio Interface]
    B --> C[Prompt Generator]
    C --> D[LLM (Zephyr-7B)]
    D --> E[gTTS: Text-to-Speech]
    E --> F[Audio Output Player]

