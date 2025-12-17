"""
Streamlit-based RAG Application for Document Retrieval.

Run with: streamlit run app.py
"""
import streamlit as st
from rag.agent import build_retrieval_agent
from vertexai.agent_engines import AdkApp
import vertexai
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()


@st.cache_resource
def initialize_agent():
    """Initialize and cache the agent."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    vertexai.init(project=project, location=location)
    
    agent = build_retrieval_agent()
    app = AdkApp(agent=agent)
    return app


def extract_response_text(response):
    """Extract text from response dict."""
    if response and isinstance(response, dict):
        if 'parts' in response:
            for part in response['parts']:
                if 'text' in part:
                    return part['text']
        elif 'content' in response and 'parts' in response['content']:
            for part in response['content']['parts']:
                if 'text' in part:
                    return part['text']
    return None


def main():
    st.set_page_config(
        page_title="Document Retrieval RAG App",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Document Retrieval RAG Application")
    st.markdown("Ask questions about your documents using AI-powered retrieval")
    
    # Initialize agent
    with st.spinner("Initializing agent..."):
        app = initialize_agent()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This RAG application uses:
        - **Vertex AI RAG Engine** for document retrieval
        - **Gemini 2.0 Flash** for response generation
        - **Google ADK** for agent orchestration
        
        Ask questions about the uploaded documents and get cited answers.
        """)
        
        st.divider()
        
        st.header("💡 Example Questions")
        
        st.subheader("📊 Financial Questions")
        financial_questions = [
            "What was Google's total revenue in Q3 2024?",
            "How did Google Cloud perform?",
            "What were the main operating expenses?",
            "What is the effective tax rate?"
        ]
        for idx, question in enumerate(financial_questions):
            if st.button(question, key=f"financial_{idx}", use_container_width=True):
                st.session_state.example_query = question
        
        st.subheader("🤖 AI/ML Research")
        ml_questions = [
            "What is retrieval-augmented generation (RAG)?",
            "Explain transformer architectures",
            "What are the best practices for MLOps?",
            "How does attention mechanism work?"
        ]
        for idx, question in enumerate(ml_questions):
            if st.button(question, key=f"ml_{idx}", use_container_width=True):
                st.session_state.example_query = question
        
        st.subheader("📚 General")
        general_questions = [
            "Summarize the main topics in the documents",
            "What are the key findings about machine learning?",
            "Compare different approaches discussed"
        ]
        for idx, question in enumerate(general_questions):
            if st.button(question, key=f"general_{idx}", use_container_width=True):
                st.session_state.example_query = question
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input (always show)
    prompt = st.chat_input("Ask a question about your documents...")
    
    # Handle sidebar example button clicks
    if "example_query" in st.session_state:
        prompt = st.session_state.example_query
        del st.session_state.example_query
    
    # Process the query (from either chat input or example button)
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Thinking..."):
                try:
                    for response in app.stream_query(message=prompt, user_id="streamlit_user"):
                        text = extract_response_text(response)
                        if text:
                            full_response += text
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    message_placeholder.error(error_msg)
                    full_response = error_msg
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
