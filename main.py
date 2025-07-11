import streamlit as st
import uuid
import time
from utils.youtube_utils import extract_video_id, fetch_transcript
from utils.rag_utils import process_transcript
from utils.chain_utils import build_rag_chain

# Page configuration
st.set_page_config(page_title="YouTube RAG Assistant", layout="wide")
st.title("🎥 YouTube Video Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# YouTube video input
url = st.text_input("Enter YouTube video URL:")

if url:
    video_id = extract_video_id(url)
    if video_id:
        video_col, chat_col = st.columns([1, 1])

        with video_col:
            st.subheader("📺 Video")
            st.video(f"https://www.youtube.com/embed/{video_id}")
            if st.button("📥 Process Video"):
                with st.spinner("Processing transcript..."):
                    try:
                        transcript = fetch_transcript(video_id)
                        retriever = process_transcript(transcript, video_id)
                        session_id = str(uuid.uuid4())
                        st.session_state.rag_chain = build_rag_chain(retriever, session_id)
                        st.session_state.session_id = session_id
                        st.session_state.messages = []  # Clear previous chat
                        st.success("✅ Chatbot is ready!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        with chat_col:
            st.subheader("💬 Chat")

            if st.session_state.rag_chain:
                # Display chat history
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                # Chat input at bottom
                prompt = st.chat_input("Ask about the video...")

                if prompt:
                    # Append user message
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        response_box = st.empty()
                        full_response = ""
                        try:
                            response = st.session_state.rag_chain.invoke(
                                {"question": prompt},
                                config={"configurable": {"session_id": st.session_state.session_id}}
                            )
                            # Simulate streaming response
                            for word in response.split():
                                full_response += word + " "
                                time.sleep(0.03)
                                response_box.markdown(full_response + "▌")
                            response_box.markdown(full_response)
                        except Exception as e:
                            full_response = f"❌ Error: {e}"
                            response_box.error(full_response)

                    # Append assistant message
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.info("Please process a video first to enable chat.")
    else:
        st.warning("⚠️ Invalid YouTube URL")
else:
    st.info("Paste a YouTube URL to get started.")
