import streamlit as st
from utils.youtube_utils import extract_video_id, fetch_transcript
from utils.rag_utils import process_transcript
from utils.chain_utils import build_rag_chain
import uuid

## Page config
st.set_page_config(page_title="YouTube RAG Assistant", layout="wide")

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Header
st.title("🎥 YouTube Video Assistant")

# YouTube URL input
url = st.text_input("Enter YouTube video URL:")

if url:
    video_id = extract_video_id(url)
    if video_id:
        video_col, chat_col = st.columns([1, 1])

        with video_col:
            st.subheader("📺 Video")
            st.video(f"https://www.youtube.com/embed/{video_id}")
            if st.button("Process Video"):
                with st.spinner("Processing transcript..."):
                    try:
                        transcript = fetch_transcript(video_id)
                        retriever = process_transcript(transcript, video_id)
                        # Generate a unique session ID
                        st.session_state.session_id = str(uuid.uuid4())
                        chain = build_rag_chain(retriever, st.session_state.session_id)
                        st.session_state.rag_chain = chain
                        st.session_state.messages = []
                        st.success("Chatbot is ready!")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with chat_col:
            st.subheader("💬 Chat")

            if st.session_state.rag_chain and st.session_state.session_id:
                # Display all chat messages (top-down)
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                # Chat input at the bottom
                user_input = st.chat_input("Ask about the video...")

                if user_input:
                    # Save user message
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    with st.chat_message("user"):
                        st.markdown(user_input)

                    try:
                        # Get assistant response with session_id in config
                        response = st.session_state.rag_chain.invoke(
                            {"question": user_input},
                            config={"configurable": {"session_id": st.session_state.session_id}}
                        )
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.markdown(response)
                    except Exception as e:
                        err_msg = f"❌ Error: {e}"
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
                        with st.chat_message("assistant"):
                            st.error(err_msg)
            else:
                st.info("Process a video first to enable the chat.")
    else:
        st.error("Invalid YouTube URL.")
else:
    st.info("Paste a YouTube URL to get started.")