import streamlit as st
import time
from dotenv import load_dotenv
import os

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Tools & Agents
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Load env vars (Groq API key, OpenAI key, etc.)
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt for RAG
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question:{input}
    """
)

# Create Vector Embedding
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # Data ingestion
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )

# Streamlit App
st.title("Hybrid RAG Q&A with Confidence Threshold + Tools")

user_prompt = st.text_input("Enter your query")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

# Define Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Initialize Agent with tools
tools = [arxiv, wiki, search]
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

if user_prompt:
    # Retrieve top docs with similarity scores
    retriever = st.session_state.vectors.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs_with_scores = st.session_state.vectors.similarity_search_with_score(user_prompt, k=3)

    # Confidence threshold
    threshold = 0.75  # tune based on embeddings
    relevant_docs = [doc for doc, score in docs_with_scores if score > threshold]

    if relevant_docs:
        # If above threshold → use RAG
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        elapsed = time.process_time() - start

        st.write(f"✅ Answer (RAG in {elapsed:.2f}s):")
        st.write(response['answer'])

        # Show supporting docs
        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('------------------------')

    else:
        # If below threshold → use external tools
        st.write("❌ Low confidence in local DB. Using external tools...")
        with st.spinner("Searching Arxiv/Wikipedia/Web..."):
            agent_response = agent.run(user_prompt, callbacks=[StreamlitCallbackHandler(st.container())])
            st.write(agent_response)







