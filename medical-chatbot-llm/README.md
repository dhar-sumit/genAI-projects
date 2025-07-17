# 🩺 Medical Chatbot with LLM (End-to-End Project)

This project is an end-to-end medical chatbot powered by a large language model (LLM). It is designed to provide **informational** responses about health-related topics without diagnosing or prescribing treatment.

## 🎯 Goal

To build a responsible, interactive medical assistant using LLMs (like GPT-4) and Streamlit, exploring GenAI in a real-world use case.

## 📦 Stack

- Python 3.10+
- OpenAI API (GPT-4)
- Flask (web UI)
- Prompt engineering
- LangChain (for advanced RAG)
- Pinecone Vector Database for retrieval
- AWS (Simple / CICD)

## 📁 Structure

| Folder/File         | Description |
|---------------------|-------------|
| `app/`              | Contains Streamlit app and logic |
| `prompts/`          | Holds system/instruction prompts |
| `notebooks/`        | Experiments and prompt testing |
| `requirements.txt`  | Python dependencies |
| `notes.md`          | Personal development notes |

## ✅ Features

- Interactive Q&A UI
- Safe, ethical response system (no diagnosis)
- Modular and extensible
- Clean prompt handling
- Ready for enhancement with LangChain or RAG

## 🔧 Setup Instructions

1. Clone the repo:
```bash
git clone https://github.com/your-username/genAI-projects.git
cd genAI-projects/medical-chatbot-llm
