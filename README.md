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
User Query
   │
   ▼
 Check Vector Similarity (FAISS)
   ├── High Confidence → RAG Answer (contextual docs)
   └── Low Confidence → Use Tools (Arxiv, Wikipedia, DuckDuckGo)
