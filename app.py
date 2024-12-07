import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL context for unverified HTTPS requests
ssl._create_default_https_context = ssl._create_unverified_context

# NLTK data path and download
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def main():
    global counter
    
    # Set background color and styles for Streamlit app
    st.markdown(
        <style>
        body {
            background-color: #F5F5DC; /* Beige for the main background */
            font-family: 'Arial', sans-serif;
        }
        .reportview-container {
            background-color: #FFFFFF; /* White for the main container */
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }
        h1 {
            color: #FF9933; /* Saffron for headers */
            text-align: center;
        }
        .sidebar .sidebar-content {
            background-color: #27408B; /* Deep Blue for sidebar */
            color: white;
        }
        .stTextInput>div>div>input {
            border-color: #FFD700; /* Gold for input border */
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }
        .stButton>button {
            background-color: #008000; /* Green for buttons */
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #800080; /* Purple on hover */
        }
        .stExpander {
            background-color: #C0C0C0; /* Silver for expanders */
            border-radius: 5px;
            padding: 10px;
        }
        .important-alert {
            background-color: #8B0000; /* Maroon for important alerts */
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        , 
        unsafe_allow_html=True
    )

    st.title("Bhagavad Gita Chatbot using NLP")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the Bhagavad Gita Chatbot. Please type a message and press Enter to start the conversation.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User  Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot _response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.text(f":User   {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found.")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on the teachings of the Bhagavad Gita. The chatbot is built using Natural Language Processing (NLP) techniques and Logistic Regression to extract intents and provide meaningful responses related to the text of the Gita. The interface is developed using Streamlit, a Python library for building interactive web applications.")

        st.subheader("Project Overview:")

        st.write("""
        The project is divided into two main parts:
        1. **NLP Techniques and Logistic Regression**: These are used to train the chatbot on labeled intents derived from the verses of the Bhagavad Gita, allowing it to understand user queries and respond appropriately.
        2. **Streamlit Chatbot Interface**: This web-based interface allows users to input their questions or thoughts and receive responses that reflect the wisdom and teachings of the Bhagavad Gita.
        """)

        st.subheader("Dataset:")

        st.write("""
        The dataset used in this project consists of labeled intents and responses based on the teachings of the Bhagavad Gita. The data is structured as follows:
        - **Intents**: The intent of the user input (e.g., "greeting", "philosophy", "life lessons").
        - **Responses**: The responses are derived from the verses of the Bhagavad Gita, providing insights and guidance on various aspects of life.
        - **Text**: The user input text, which the chatbot interprets to provide relevant responses.
        """)

        st.subheader("Streamlit Chatbot Interface:")

        st.write("The chatbot interface is built using Streamlit, featuring a text input box for users to ask questions and a chat window to display the chatbot's responses. The interface leverages the trained model to generate responses that reflect the philosophical teachings of the Bhagavad Gita.")

        st.subheader("Conclusion:")

        st.write("In this project, a chatbot is developed to help users explore the profound teachings of the Bhagavad Gita. By utilizing NLP and Logistic Regression, the chatbot can understand and respond to user inquiries, making the wisdom of the Gita accessible to a wider audience. This project can be further enhanced by incorporating more verses, expanding the dataset, and utilizing advanced NLP techniques.")
if __name__ == '__main__':
    main()









# import os
# import json
# import datetime
# import csv
# import nltk
# import ssl
# import streamlit as st
# import random
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# ssl._create_default_https_context = ssl._create_unverified_context
# nltk.data.path.append(os.path.abspath("nltk_data"))
# nltk.download('punkt')

# # Load intents from the JSON file
# file_path = os.path.abspath("./intents.json")
# with open(file_path, "r") as file:
#     intents = json.load(file)

# # Create the vectorizer and classifier
# vectorizer = TfidfVectorizer(ngram_range=(1, 4))
# clf = LogisticRegression(random_state=0, max_iter=10000)

# # Preprocess the data
# tags = []
# patterns = []
# for intent in intents:
#     for pattern in intent['patterns']:
#         tags.append(intent['tag'])
#         patterns.append(pattern)

# # training the model
# x = vectorizer.fit_transform(patterns)
# y = tags
# clf.fit(x, y)

# def chatbot(input_text):
#     input_text = vectorizer.transform([input_text])
#     tag = clf.predict(input_text)[0]
#     for intent in intents:
#         if intent['tag'] == tag:
#             response = random.choice(intent['responses'])
#             return response
        
# counter = 0

# def main():
#     global counter
    
#     # Set background color and styles for Streamlit app
#     st.markdown(
#         """
#         <style>
#         body {
#             background-color: #F5F5DC; /* Beige for the main background */
#             font-family: 'Arial', sans-serif;
#         }
#         .reportview-container {
#             background-color: #FFFFFF; /* White for the main container */
#             border-radius: 10px;
#             padding: 20px;
#             box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
#         }
#         h1 {
#             color: #FF9933; /* Saffron for headers */
#             text-align: center;
#         }
#         .sidebar .sidebar-content {
#             background-color: #27408B; /* Deep Blue for sidebar */
#             color: white;
#         }
#         .stTextInput>div>div>input {
#             border-color: #FFD700; /* Gold for input border */
#             border-radius: 5px;
#             padding: 10px;
#             font-size: 16px;
#         }
#         .stButton>button {
#             background-color: #008000; /* Green for buttons */
#             color: white;
#             border-radius: 5px;
#             padding: 10px 20px;
#             font-size: 16px;
#             transition: background-color 0.3s;
#         }
#         .stButton>button:hover {
#             background-color: #800080; /* Purple on hover */
#         }
#         .stExpander {
#             background-color: #C0C0C0; /* Silver for expanders */
#             border-radius: 5px;
#             padding: 10px;
#         }
#         .important-alert {
#             background-color: #8B0000; /* Maroon for important alerts */
#             color: white;
#             padding: 10px;
#             border-radius: 5px;
#         }
#         </style>
#         """, 
#         unsafe_allow_html=True
#     )

#     st.title("Bhagavad Gita Chatbot using NLP")

#     # Create a sidebar menu with options
#     menu = ["Home", "Conversation History", "About"]
#     choice = st.sidebar.selectbox("Menu", menu)

#     # Home Menu
#     if choice == "Home":
#         st.write("Welcome to the Bhagavad Gita Chatbot. Please type a message and press Enter to start the conversation.")

#         # Check if the chat_log.csv file exists, and if not, create it with column names
#         if not os.path.exists('chat_log.csv'):
#             with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
#                 csv_writer = csv.writer(csvfile)
#                 csv_writer.writerow(['User  Input', 'Chatbot Response', 'Timestamp'])

#         counter += 1
#         user_input = st.text_input("You:", key=f"user_input_{counter}")

#         if user_input:
#             # Convert the user input to a string
#             user_input_str = str(user_input)

#             response = chatbot(user_input)
#             st.text_area("Chatbot :", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

#             # Get the current timestamp
#             timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

#             # Save the user input and chatbot response to the chat_log.csv file
#             with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
#                 csv_writer = csv.writer(csvfile)
#                 csv_writer.writerow([user_input_str, response, timestamp])

#             if response.lower() in ['goodbye', 'bye']:
#                 st.write("Thank you for chatting with me. Have a great day!")
#                 st.stop()

#     # Conversation History Menu
#     elif choice == "Conversation History":
#         # Display the conversation history in a collapsible expander
#         st.header("Conversation History")
#         with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
#             csv_reader = csv.reader(csvfile)
#             next(csv_reader)  # Skip the header row
#             for row in csv_reader:
#                 st.text(f":User  {row[0]}")
#                 st.text(f"Chatbot: {row[1]}")
#                 st.text(f"Timestamp: {row[2]}")
#                 st.markdown("---")

#     elif choice == "About":
#         st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

#         st.subheader("Project Overview:")

#         st.write("""
#         The project is divided into two parts:
#         1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
#         2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
#         """)

#         st.subheader("Dataset:")

#         st.write("""
#         The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
#         - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
#         - Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
#         - Text: The user input text.
#         """)

#         st.subheader("Streamlit Chatbot Interface:")

#         st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

#         st.subheader("Conclusion:")

#         st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")

# if __name__ == '__main__':
#     main()
