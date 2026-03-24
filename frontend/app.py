import streamlit as st
import requests
import time
import io
from datetime import datetime

SL_API_URL = "http://localhost:8000/api/query/"


st.set_page_config(
    page_title="RAG Chat",
    page_icon=":cyclone:",
    layout="centered",
    initial_sidebar_state="expanded"
)


if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize settings
if "max_results" not in st.session_state:
    st.session_state.max_results = 5

if "include_sources" not in st.session_state:
    st.session_state.include_sources = True
    
def query_api(question: str, max_results: int = 5, include_sources: bool = True) -> dict:

    payload = {
        "question": question,
        "max_results": max_results,
        "include_sources": include_sources
    }
    
    try:
        response = requests.post(
            SL_API_URL,
            json=payload,
            timeout=30  
        )
        response.raise_for_status() 
        return response.json()
        
    except requests.exceptions.ConnectionError:
        raise Exception("Cannot connect to API. Is the backend running at " + SL_API_URL + "?")
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. The query might be too complex.")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")
    

st.title("RAG Multi-Agent Document Intelligence")
st.markdown("Ask questions about your documents using our multi-agent RAG system.")

chatTab, ingestTab = st.tabs(["Chat", "Upload Documents"])


with chatTab:
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    with st.sidebar:
        st.title("Settings")
        

        # Toggle sources
        st.session_state.include_sources = st.checkbox(
            "Show Sources",
            value=st.session_state.include_sources,
            help="Display source documents in responses"
        )
        
        st.divider()
        

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        st.markdown("### About")
        st.markdown(
            """
            This is a multi-agent RAG system powered by:
            - a Query Agent
            - a Retrieval Analysis Agent
            - a Response Agent
            
            Built with LangGraph, FastAPI.
            """
        )
        
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
    
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # User message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI Chat Message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = query_api(
                        question=prompt,
                        max_results=st.session_state.max_results,
                        include_sources=st.session_state.include_sources
                    )
                    
                    answer = response.get("answer", "No answer generated")
                    confidence = response.get("confidence_score", 0.0)
                    quality = response.get("retrieval_quality", "unknown")
                    sources = response.get("sources", [])
                    processing_time = response.get("processing_time", 0.0)
                    
                    st.markdown(answer)
                    
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {
                            "confidence_score": confidence,
                            "retrieval_quality": quality,
                            "sources": sources,
                            "processing_time": processing_time
                        }
                    })
                    
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })

with ingestTab: 
    st.header("Upload Documents")
    st.markdown("Upload documents to add them to the knowledge base.")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "pdf", "csv"],
        help="Supported formats: TXT, PDF, CSV"
    )
    
if uploaded_file is not None:
        st.info(f"**File:** {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        if st.button("Upload and Ingest", type="primary", use_container_width=True):
            with st.spinner("Uploading and processing..."):
                try:
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    
                    upload_url = SL_API_URL.replace("/query/", "/documents/upload")
                    response = requests.post(
                        upload_url,
                        files=files,
                        timeout=60 
                    )
                    
                    print(response)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"{data['message']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Chunks Created", data['chunks_created'])
                        with col2:
                            st.metric("Total Documents", data['total_documents'])
                    else:
                        st.error(f"Upload failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
st.divider()

# List existing documents
st.subheader("Current Documents")

if st.button("Refresh List"):
    st.rerun()

try:
    list_url = SL_API_URL.replace("/query/", "/documents/list")
    response = requests.get(list_url, timeout=5)
    
    if response.status_code == 200:
        data = response.json()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", data['total_chunks'])
        with col2:
            st.metric("Unique Documents", data['unique_documents'])
        
        # Show document list
        if data['documents']:
            st.markdown("**Documents in system:**")
            for i, doc in enumerate(data['documents'], 1):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"{i}. {doc}")
                with col2:
                    
                    if st.button("Delete", key=f"delete_{doc}"):
                        delete_url = SL_API_URL.replace("/query/", f"/documents/{doc}")
                        try:
                            del_response = requests.delete(delete_url, timeout=10)
                            if del_response.status_code == 200:
                                st.success(f"Deleted {doc}")
                                st.rerun()
                            else:
                                st.error(f"Error: {del_response.text}")
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.info("No documents in system yet.")
    else:
        st.error("Cannot load document list")
        
except Exception as e:
    st.error(f"Error: {str(e)}")