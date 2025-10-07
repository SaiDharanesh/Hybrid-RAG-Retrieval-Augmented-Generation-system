Hybrid RAG Q&A System with Confidence Threshold + Tools

This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, Groq’s Llama3 model, and external knowledge sources like Arxiv, Wikipedia, and DuckDuckGo Search.

If a user’s question is relevant to local documents, the system answers using the vector database (RAG).
If not, it automatically switches to external tools for real-time web or academic retrieval — achieving a hybrid intelligence workflow.

🚀 Features

✅ RAG-based Q&A using FAISS Vector Store
✅ Confidence Thresholding to decide between internal vs external knowledge
✅ External Tool Integration (Arxiv, Wikipedia, DuckDuckGo Search)
✅ Groq-powered LLM (Llama3-8b) for fast, low-latency responses
✅ Streamlit UI for interactive exploration
✅ Automatic document ingestion and embedding creation

🧩 Architecture Overview
<img width="1024" height="1536" alt="Hybrid Question Answering System Flowchart" src="https://github.com/user-attachments/assets/0c25f303-9c6a-4a71-a259-ebe769269268" />

