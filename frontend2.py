# frontend.py
import streamlit as st

import requests
import json

# URL for your Flask backend (running on localhost by default)
FLASK_URL =  "http://localhost:5002/generate" # Adjust the Flask server URL if running on a different host/port

# Streamlit UI setup
st.title("Target Audience Knowlege Agent with LangGraph")
st.subheader(':blue[Welcome! This tool allows you to interact with Audience Knowledage Bank and Websearch ]')

st.markdown("Specific topics are strategies to find the target audience for your brands and why identifying your target audience is important to your marketing strategy.")
st.markdown("links to documents: \n\n https://www.adroll.com/blog/5-tools-to-learn-about-your-target-audience, https://online.hbs.edu/blog/post/target-audience-in-marketing, https://www.adobe.com/express/learn/blog/target-audience")
# Text input for user question
st.subheader("Ask a Question")


# Create a button to trigger question handling

# if st.button("Submit Question"):
#     if question.strip():
#         with st.spinner("Generating response..."):
#             # Make a POST request to the Flask backend
#             try:
#                 response = requests.post(f"{FLASK_URL}", json={"question": question})
#                 if response.status_code == 200:
#                     data = response.json()
#                     results = data.get('results', [])
                    
#                     # Display each key-value pair
#                     for result in results:
#                         for key, value in result.items():
#                             st.markdown(f"**{key}:** {value}")
                    
#                     # Display the final response
#                     st.markdown(f"### **Response:** {data.get('response', 'No response')}")

#                 else:
#                     st.error(f"Error: {response.json().get('error', 'Unknown error')}")

#             except Exception as e:
#                 st.error(f"Connection error: {e}")
#     else:
#         st.warning("Please enter a question.")



if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []  # Initialize chat history
#question = st.text_input("Enter your question:")
# if st.button("Submit Question"):
#     if  question.strip():
#         # Append the user's question to chat history
#         st.session_state['chat_history'].append({"role": "human", "content": question})
        
#         with st.spinner("Generating response..."):
#             # Prepare payload with the chat history
#             try:
#                 response = requests.post(
#                     f"{FLASK_URL}",
#                     json={"messages": st.session_state['chat_history']}
#                 )
#                 if response.status_code == 200:
#                     data = response.json()
#                     ai_response = data.get('response', 'No response')

#                     # Append the AI's response to chat history
#                     st.session_state['chat_history'].append({"role": "AI", "content": ai_response})

#                     # Display the chat history
#                     for message in st.session_state['chat_history']:
#                         if message["role"] == "human":
#                             st.markdown(f"**Human:** {message['content']}")
#                         elif message["role"] == "AI":
#                             st.markdown(f"**AI:** {message['content']}")

#                 else:
#                     st.error(f"Error: {response.json().get('error', 'Unknown error')}")

#             except Exception as e:
#                 st.error(f"Connection error: {e}")
#     else:
#         st.warning("Please enter a question.")
chat_container = st.container()
with chat_container:
    for message in st.session_state['chat_history']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
question = st.text_input("Enter your question:")
if st.button("Submit Question"):
    if question.strip():
   #     st.session_state['chat_history'].append({"role": "human", "content": question})

     #   with st.spinner("Generating response..."):
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
                
              #  response = requests.post(f"{FLASK_URL}", json={"question": question})
              #  if response.status_code == 200:
              #      data = response.json()
              #      results = data.get('results', [])
              #      AI_response=data.get('response', 'No response')
                   # st.session_state['chat_history'].append({"role": "AI", "content": AI_response})
                    # Display each key-value pair
                 #   for result in results:
                 #       for key, value in result.items():
                 #           st.markdown(f"**{key}:** {value}")
               
          #          st.session_state['history'].append({"role": "user", "content": question})
                    
       #             with st.chat_message("assistant"):
       #                 with st.spinner("Generating response..."):
       #                 response = query_engine.query(user_input)
       #                 full_response = response.response
       #                 st.markdown(full_response)
                    
       #             for message in st.session_state['chat_history']:
                        #st.markdown(message)
      #                  if message["role"] == "human":
       #                      st.markdown(f"**Human:** {message['content']}")
       #                 elif message["role"] == "AI":
       #                      st.markdown(f"**AI:** {message['content']}")

                    # Display the final response
              #      st.markdown(f"### **Response:** {data.get('response', 'No response')}")

                #else:
                #    st.error(f"Error: {response.json().get('error', 'Unknown error')}")

            except Exception as e:
                st.error(f"Connection error: {e}")
    else:
        st.warning("Please enter a question.")

if st.button("Clear Chat"):
    st.session_state['chat_history'] = []
    st.rerun()