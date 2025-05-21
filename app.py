import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
import time

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
    
    # Create tabs in sidebar - removed the About tab
    sidebar_tab1, sidebar_tab2 = st.tabs(["Settings", "Advanced"])
    
    with sidebar_tab1:
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
    
    with sidebar_tab2:
        # Moved from tab1 to tab2: Focus options
        summary_focus = st.multiselect(
            "Focus On:",
            ["Key Facts", "Statistics", "Quotes", "Background Context", "Future Implications"],
            default=["Key Facts"]
        )
        
        # Advanced Options
        st.subheader("Advanced Options")
        
        # Set max_length to 0 (no limit) directly in the code
        max_length = 0
        
        # Text cleaning options as tickmarks (radio buttons)
        col1, col2 = st.columns(2)
        
        with col1:
            clean_whitespace = st.radio(
                "Clean whitespace:",
                options=["Off", "On"],
                index=1  # Default to "On"
            )
            
            remove_urls = st.radio(
                "Remove URLs:",
                options=["Off", "On"],
                index=1  # Default to "On"
            )
        
        with col2:
            remove_html = st.radio(
                "Remove HTML tags:",
                options=["Off", "On"],
                index=1  # Default to "On"
            )

with tab1:
    # Input options
    input_option = st.radio("Input Type:", ["Paste Text", "Upload File", "URL"], horizontal=True)
    
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
    
    elif input_option == "URL":
        url = st.text_input("Enter news article URL:")
        if url:
            with st.spinner("Extracting content from URL..."):
                try:
                    # Add http:// if not present
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                        
                    # Enhanced headers to mimic a real browser
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none',
                        'Sec-Fetch-User': '?1',
                        'Cache-Control': 'max-age=0',
                        'DNT': '1'
                    }
                    
                    # Add retry mechanism with exponential backoff
                    max_retries = 3
                    base_delay = 2  # seconds
                    
                    for attempt in range(max_retries):
                        try:
                            # Create a session for persistent cookies
                            session = requests.Session()
                            
                            # First, try to get the main page
                            response = session.get(url, headers=headers, timeout=15)
                            response.raise_for_status()
                            
                            # Parse the content
                            soup = BeautifulSoup(response.content, 'html.parser')
                            
                            # Remove unwanted elements
                            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'noscript', 'meta', 'link', 'aside']):
                                element.decompose()
                            
                            # Special handling for AI news websites
                            if 'artificialintelligence-news.com' in url:
                                # Try to find the main content area
                                main_content = soup.find('div', class_='td-main-content') or \
                                             soup.find('div', class_='td-ss-main-content') or \
                                             soup.find('article')
                                
                                if main_content:
                                    # Get all article elements
                                    articles = main_content.find_all(['article', 'div'], class_=lambda x: x and ('post' in str(x).lower() or 'article' in str(x).lower()))
                                    
                                    if articles:
                                        article_text = ""
                                        for article in articles:
                                            # Get the title
                                            title = article.find(['h1', 'h2', 'h3'])
                                            if title:
                                                article_text += title.get_text(strip=True) + "\n\n"
                                            
                                            # Get the content
                                            content = article.find(['div', 'section'], class_=lambda x: x and ('content' in str(x).lower() or 'entry' in str(x).lower()))
                                            if content:
                                                article_text += content.get_text(separator='\n', strip=True) + "\n\n"
                                        
                                        if article_text:
                                            break
                            
                            # If no special handling worked, try general selectors
                            if not article_text:
                                article_selectors = [
                                    'article', '.article-body', '.entry-content', 
                                    '.story-body', '.post-content', '.news-content',
                                    'main', '.main-content', '#content', '.content',
                                    '.article-content', '.post-body', '.story-content',
                                    '[role="main"]', '.article', '.story',
                                    '.td-main-content', '.td-ss-main-content'
                                ]
                                
                                for selector in article_selectors:
                                    article_elements = soup.select(selector)
                                    if article_elements:
                                        for elem in article_elements:
                                            article_text += elem.get_text(separator='\n', strip=True) + "\n"
                                        break
                            
                            # If still no content, try to get main content
                            if not article_text:
                                main_content = soup.find('main') or \
                                             soup.find('article') or \
                                             soup.find('div', class_=lambda x: x and ('content' in str(x).lower() or 'article' in str(x).lower() or 'post' in str(x).lower()))
                                
                                if main_content:
                                    article_text = main_content.get_text(separator='\n', strip=True)
                                else:
                                    article_text = soup.body.get_text(separator='\n', strip=True)
                            
                            # Clean up the text
                            import re
                            # Remove excessive whitespace and newlines
                            article_text = re.sub(r'\n\s*\n', '\n\n', article_text)
                            # Remove very short lines (likely navigation or ads)
                            article_text = '\n'.join(line for line in article_text.split('\n') if len(line.strip()) > 20)
                            # Remove common non-content text
                            article_text = re.sub(r'(?i)(privacy policy|terms of use|cookie policy|all rights reserved).*$', '', article_text, flags=re.MULTILINE)
                            
                            news_text = article_text.strip()
                            
                            if news_text:
                                break
                                
                        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as e:
                            if attempt == max_retries - 1:  # Last attempt
                                raise e
                            delay = base_delay * (2 ** attempt)  # Exponential backoff
                            st.warning(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
                            time.sleep(delay)
                    
                    # Display a preview
                    if news_text:
                        st.success("Content extracted successfully!")
                        with st.expander("Preview extracted content"):
                            st.text(news_text[:1000] + "..." if len(news_text) > 1000 else news_text)
                    else:
                        st.warning("Could not extract meaningful content from the URL. Please try pasting the content manually.")
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching content: {str(e)}")
                    st.info("Please check the URL or paste the content manually.")
                except Exception as e:
                    st.error(f"Error processing content: {str(e)}")
                    st.info("Please try pasting the content manually.")
    
    # Process button
    if st.button("Generate Summary"):
        if news_text:
            # Apply preprocessing if needed
            if max_length > 0 and len(news_text) > max_length:
                news_text = news_text[:max_length]
                st.info(f"Text truncated to {max_length} characters for processing.")
            
            # Convert select_slider values to boolean
            if remove_urls == "On":
                import re
                news_text = re.sub(r'http\S+', '', news_text)
            
            if clean_whitespace == "On":
                import re
                news_text = re.sub(r'\s+', ' ', news_text).strip()
            
            if remove_html == "On":
                from bs4 import BeautifulSoup
                news_text = BeautifulSoup(news_text, "html.parser").get_text()
                
            # Create session state for history if not exists
            if "history" not in st.session_state:
                st.session_state.history = []
                
            try:
                # Create message placeholder
                response_placeholder = st.empty()
                
                # Create a stream handler
                stream_handler = StreamHandler(response_placeholder)
                
                # Get streaming model with handler
                chat = get_streaming_chat_model(stream_handler)
                
                # Create prompt based on user selections
                focus_points = ", ".join(summary_focus) if summary_focus else "Key Facts"
                
                prompt = f"""
                You are a professional news summarizer. Summarize the following news article in {output_language}.
                
                Article language: {input_language}
                Requested summary length: {summary_length}
                Summary style: {summary_style}
                Focus on: {focus_points}
                
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
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Option to regenerate with different settings
                    if st.button("Regenerate with current settings", key=f"regen_{i}"):
                        # Copy the text to the main tab
                        st.session_state.regenerate_text = item["original_text"]
                        st.rerun()
                with col2:
                    # Option to download this summary
                    st.download_button(
                        label="Download Summary",
                        data=f"ORIGINAL TEXT ({item['input_language']}):\n\n{item['original_text']}\n\nSUMMARY ({item['output_language']}):\n\n{item['summary']}",
                        file_name=f"summary_{len(st.session_state.history) - i}.txt",
                        mime="text/plain",
                        key=f"dl_{i}"
                    )
                with col3:
                    # Option to translate to another language
                    if st.button("Translate to new language", key=f"trans_{i}"):
                        # Set session state to indicate translation is needed
                        st.session_state.translate_item = item
                        st.session_state.translating = True
                        st.rerun()
    else:
        st.info("No summaries generated yet. Use the Summarize News tab to create summaries.")
        
# Add a translation popup if needed
if "translating" in st.session_state and st.session_state.translating:
    with st.sidebar:
        st.markdown("### 🔄 Translate Summary")
        target_lang = st.selectbox("Target Language:", languages)
        
        if st.button("Translate"):
            item = st.session_state.translate_item
            
            try:
                # Create message placeholder
                with st.spinner(f"Translating to {target_lang}..."):
                    chat = get_base_chat_model()
                    
                    # Create prompt for translation
                    prompt = f"""
                    Translate the following summary from {item['output_language']} to {target_lang}:
                    
                    {item['summary']}
                    """
                    
                    # Generate translation
                    messages = [HumanMessage(content=prompt)]
                    response = chat.invoke(messages)
                    translation = response.content
                    
                    # Add to history
                    st.session_state.history.append({
                        "original_text": item["original_text"],
                        "summary": translation,
                        "input_language": item["input_language"],
                        "output_language": target_lang,
                        "length": item["length"],
                        "style": item["style"] + " (Translated)"
                    })
                    
                # Clear translation state
                st.session_state.translating = False
                del st.session_state.translate_item
                st.rerun()
                    
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
        
        if st.button("Cancel"):
            st.session_state.translating = False
            if "translate_item" in st.session_state:
                del st.session_state.translate_item
            st.rerun()

# Handle regeneration
if "regenerate_text" in st.session_state:
    # Switch to the first tab and populate with the text
    st.session_state.news_text = st.session_state.regenerate_text
    del st.session_state.regenerate_text
    # Note: The actual population has to happen in the next run
