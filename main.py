import streamlit as st
from models import get_client
from rag_helper import RAGBase

def main():
    client = get_client()

    rag = RAGBase()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.gemini_chat = client.chats.create(
            model='gemini-3-flash-preview'
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask me anything about the German Laws regarding registration."):
        with st.chat_message("user"):
            st.markdown(query)
        
        st.session_state.messages.append({"role": "user", "content": query})

        final_prompt = rag.get_context(query)

        response = st.session_state.gemini_chat.send_message(final_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        

if __name__ == "__main__":
    main()