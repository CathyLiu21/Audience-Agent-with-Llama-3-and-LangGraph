# frontend.py
import streamlit as st

import requests
import json

# URL for your Flask backend (running on localhost by default)
FLASK_URL =  "http://localhost:5002/generate" # Adjust the Flask server URL if running on a different host/port

############## Streamlit UI setup
st.title("Target Audience Knowlege Agent with LangGraph")
st.subheader(':blue[Welcome! This tool allows you to interact with Audience Knowledage Bank and Websearch ]')

st.markdown("Specific topics are strategies to find the target audience for your brands and why identifying your target audience is important to your marketing strategy.")
st.markdown("links to documents: \n\n https://www.adroll.com/blog/5-tools-to-learn-about-your-target-audience, https://online.hbs.edu/blog/post/target-audience-in-marketing, https://www.adobe.com/express/learn/blog/target-audience")
# Text input for user question
st.subheader("Ask a Question")



# store history
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []  # Initialize chat history

chat_container = st.container()
with chat_container:
    for message in st.session_state['chat_history']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

# ask a question
question = st.text_input("Enter your question:")
if st.button("Submit Question"):
    if question.strip():

            # Make a POST request to the Flask backend
            try:
                with st.chat_message("user"):
                     st.markdown(question)
                st.session_state['chat_history'].append({"role": "user", "content": question})
                
                with st.chat_message("assistant"):
                    with st.spinner("Generating response..."):
                        response =requests.post(f"{FLASK_URL}", json={"question": question})
                        if response.status_code == 200:
                           data = response.json()
                           results = data.get('results', [])
                           AI_response=data.get('response', 'No response')
                           st.markdown(AI_response)
                st.session_state['chat_history'].append({"role": "assistant", "content": AI_response})
                st.rerun()  # Force a rerun to update the chat display
        

            except Exception as e:
                st.error(f"Connection error: {e}")
    else:
        st.warning("Please enter a question.")

if st.button("Clear Chat"):
    st.session_state['chat_history'] = []
    st.rerun()
