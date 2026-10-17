import streamlit as st
from models_config import get_client
from rag_helper import  RAGBaseQdrant
import os
from dotenv import load_dotenv

# def main():
#     # client = get_client()

#     rag = RAGBaseQdrant()

#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#         st.session_state.gemini_chat = client.chats.create(
#             model='gemini-3-flash-preview'
#         )

#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     if query := st.chat_input("Ask me anything about the German Laws regarding registration."):
#         with st.chat_message("user"):
#             st.markdown(query)
        
#         st.session_state.messages.append({"role": "user", "content": query})

#         final_prompt = rag.get_context(query)

#         response = st.session_state.gemini_chat.send_message(final_prompt)
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 st.markdown(response.text)
#         st.session_state.messages.append({"role": "assistant", "content": response.text})
        

# if __name__ == "__main__":
#     main()


def main():
    st.title("German Rules Assistant")
    load_dotenv()

    # Pass the name of the prompt registered in MLflow
    rag = RAGBaseQdrant(
        prompt_name="german_law_prompt", prompt_version=1
    )

    gemini_client = get_client(api_key=os.getenv("GEMINI_API_KEY"))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "gemini_chat" not in st.session_state:
        st.session_state.gemini_chat = gemini_client.chats.create(
            model="gemini-2.5-flash"
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input(
        "Ask me anything about German Laws regarding registration..."
    ):
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # 1. Retrieve raw contexts AND build prompt template
        retrieved_docs, final_prompt = rag.get_context_and_prompt(query)

        # 2. Call Gemini
        with st.chat_message("assistant"):
            with st.spinner("Searching regulations and thinking..."):
                response = st.session_state.gemini_chat.send_message(
                    final_prompt
                )
                st.markdown(response.text)

        # 3. Store response and context history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.text,
                "retrieved_context": retrieved_docs,
            }
        )


if __name__ == "__main__":
    main()