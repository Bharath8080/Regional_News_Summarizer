import os
import streamlit as st
import PyPDF2
import io
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Regional Language News Summarizer",
    page_icon="📰",
    layout="wide"
)

# Define supported languages
languages = [
    "English", "Hindi", "Gujarati", "Bengali", "Tamil", 
    "Telugu", "Kannada", "Malayalam", "Punjabi", "Marathi", 
    "Urdu", "Assamese", "Odia", "Sanskrit", "Korean", 
    "Japanese", "Arabic", "French", "German", "Spanish", 
    "Portuguese", "Russian", "Chinese", "Vietnamese", "Thai", 
    "Indonesian", "Turkish", "Polish", "Ukrainian", "Dutch", 
    "Italian", "Greek", "Hebrew", "Persian", "Swedish", 
    "Norwegian", "Danish", "Finnish", "Czech", "Hungarian", 
    "Romanian", "Bulgarian", "Croatian", "Serbian", "Slovak", 
    "Slovenian", "Estonian", "Latvian", "Lithuanian", "Malay", 
    "Tagalog", "Swahili"
]

# Streaming callback handler
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None
    
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# Initialize the ChatOpenAI model - base instance for caching
@st.cache_resource
def get_base_chat_model():
    return ChatOpenAI(
        api_key=os.getenv("SUTRA_API_KEY"),
        base_url="https://api.two.ai/v2",
        model="sutra-v2",
        temperature=0.7,
    )

# Create a streaming version of the model with callback handler
def get_streaming_chat_model(callback_handler=None):
    # Create a new instance with streaming enabled
    return ChatOpenAI(
        api_key=os.getenv("SUTRA_API_KEY"),
        base_url="https://api.two.ai/v2",
        model="sutra-v2",
        temperature=0.7,
        streaming=True,
        callbacks=[callback_handler] if callback_handler else None
    )

# Main content area
st.markdown(
    f'<h1><img src="https://framerusercontent.com/images/9vH8BcjXKRcC5OrSfkohhSyDgX0.png" width="60" style="vertical-align: middle;"/> Regional News Summarizer <img src="https://media.baamboozle.com/uploads/images/821733/1656648869_810178_gif-url.gif" width="70" height="70" style="vertical-align: middle;"/></h1>',
    unsafe_allow_html=True
)

# Setup tabs
tab1, tab2 = st.tabs(["✍️ Summarize News", "📋 History"])

with st.sidebar:
    st.title("📰 News Summarizer")
    
    # API Key input
    st.subheader("🔑 API Settings")
    api_key = st.text_input("Sutra API Key", type="password", value=os.getenv("SUTRA_API_KEY", ""))
    st.markdown("Get your API key from [SUTRA API](https://www.two.ai/sutra/api)")
    
    if not api_key:
        st.warning("⚠️ Please enter your Sutra API key to use the chatbot.")
    else:
        os.environ["SUTRA_API_KEY"] = api_key
    
    # Settings
    st.subheader("⚙️ Settings")
    
    # Input language selector
    input_language = st.selectbox("Source Language:", languages, index=0)
    
    # Output language selector
    output_language = st.selectbox("Summary Language:", languages, index=0)
    
    # Summary length as select_slider
    summary_length = st.selectbox(
        "Summary Length:",
        options=["Very Short", "Short", "Medium", "Detailed", "Comprehensive"],
    )
    
    # Style options
    summary_style = st.selectbox(
        "Summary Style:",
        ["Neutral", "Simplified", "Academic", "Conversational", "Bullet Points"]
    )

with tab1:
    # Input options
    input_option = st.radio("Input Type:", ["Paste Text", "Upload File"], horizontal=True)
    
    news_text = ""
    
    if input_option == "Paste Text":
        news_text = st.text_area("Paste news article here:", height=300)
    
    elif input_option == "Upload File":
        uploaded_file = st.file_uploader("Choose a file:", type=["txt", "md", "pdf"])
        if uploaded_file is not None:
            # Handle text files
            if uploaded_file.type in ["text/plain", "text/markdown"]:
                news_text = uploaded_file.read().decode("utf-8")
            # Handle PDF files
            elif uploaded_file.type == "application/pdf":
                try:
                    with st.spinner("Extracting text from PDF..."):
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                        news_text = ""
                        for page_num in range(len(pdf_reader.pages)):
                            news_text += pdf_reader.pages[page_num].extract_text() + "\n"
                        
                        if not news_text.strip():
                            st.warning("Could not extract text from PDF. The file might be scanned or protected.")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
    
    # Process button
    if st.button("Generate Summary"):
        if news_text:
            try:
                # Create message placeholder
                response_placeholder = st.empty()
                
                # Create a stream handler
                stream_handler = StreamHandler(response_placeholder)
                
                # Get streaming model with handler
                chat = get_streaming_chat_model(stream_handler)
                
                # Create prompt based on user selections
                prompt = f"""
                You are a professional news summarizer. Summarize the following news article in {output_language}.
                
                Article language: {input_language}
                Requested summary length: {summary_length}
                Summary style: {summary_style}
                
                Please provide a clear, accurate summary that captures the main points of the article.
                If the article contains statistics or quotes, include the most significant ones.
                
                Article text:
                {news_text}
                """
                
                # Generate streaming response
                messages = [HumanMessage(content=prompt)]
                response = chat.invoke(messages)
                summary = response.content
                
                # Add to history
                st.session_state.history.append({
                    "original_text": news_text[:300] + "..." if len(news_text) > 300 else news_text,
                    "summary": summary,
                    "input_language": input_language,
                    "output_language": output_language,
                    "length": summary_length,
                    "style": summary_style
                })
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                if "API key" in str(e):
                    st.error("Please check your Sutra API key in the environment variables.")
        else:
            st.warning("Please enter or upload news text to summarize.")

with tab2:
    # Display history of summaries
    if "history" in st.session_state and st.session_state.history:
        # Add option to clear history
        if st.button("Clear History", type="secondary"):
            st.session_state.history = []
            st.rerun()
            
        # Add download options
        st.download_button(
            label="Download All Summaries (TXT)",
            data="\n\n".join([f"SUMMARY #{i+1}\n\nOriginal Text: {item['original_text']}\n\nSummary ({item['output_language']}): {item['summary']}" 
                            for i, item in enumerate(st.session_state.history)]),
            file_name="news_summaries.txt",
            mime="text/plain"
        )
        
        # Display individual summaries
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Summary #{len(st.session_state.history) - i}", expanded=(i==0)):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Original Text")
                    st.text(item["original_text"])
                    st.caption(f"Language: {item['input_language']}")
                    
                with col2:
                    st.markdown("#### Summary")
                    st.markdown(item["summary"])
                    st.caption(f"Language: {item['output_language']} | Style: {item['style']} | Length: {item['length']}")
                
                # Options for this summary
                col1, col2 = st.columns(2)
                with col1:
                    # Option to download this summary
                    st.download_button(
                        label="Download Summary",
                        data=f"ORIGINAL TEXT ({item['input_language']}):\n\n{item['original_text']}\n\nSUMMARY ({item['output_language']}):\n\n{item['summary']}",
                        file_name=f"summary_{len(st.session_state.history) - i}.txt",
                        mime="text/plain",
                        key=f"dl_{i}"
                    )
    else:
        st.info("No summaries generated yet. Use the Summarize News tab to create summaries.")
