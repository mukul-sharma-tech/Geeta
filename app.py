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
#     st.title("Intents of Chatbot using NLP")

#     # Create a sidebar menu with options
#     menu = ["Home", "Conversation History", "About"]
#     choice = st.sidebar.selectbox("Menu", menu)

#     # Home Menu
#     if choice == "Home":
#         st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

#         # Check if the chat_log.csv file exists, and if not, create it with column names
#         if not os.path.exists('chat_log.csv'):
#             with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
#                 csv_writer = csv.writer(csvfile)
#                 csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

#         counter += 1
#         user_input = st.text_input("You:", key=f"user_input_{counter}")

#         if user_input:

#             # Convert the user input to a string
#             user_input_str = str(user_input)

#             response = chatbot(user_input)
#             st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

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
#         # with st.beta_expander("Click to see Conversation History"):
#         with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
#             csv_reader = csv.reader(csvfile)
#             next(csv_reader)  # Skip the header row
#             for row in csv_reader:
#                 st.text(f"User: {row[0]}")
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



# # second
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
    
#     # Set background color for Streamlit app to a divine theme
#     st.markdown(
#         """
#         <style>
#         .reportview-container {
#             background-color: #f0f8ff;
#         }
#         .sidebar .sidebar-content {
#             background-color: #ffebcd;
#         }
#         h1 {
#             color: #800080;
#         }
#         .stTextInput>div>div>input {
#             border-color: #ff6347;
#         }
#         .stButton>button {
#             background-color: #ff6347;
#             color: white;
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
#                 csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

#         counter += 1
#         user_input = st.text_input("You:", key=f"user_input_{counter}")

#         if user_input:

#             # Convert the user input to a string
#             user_input_str = str(user_input)

#             response = chatbot(user_input)
#             st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

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
#                 st.text(f"User: {row[0]}")
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




# # third
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
#             background: linear-gradient(to right, #ff7e5f, #feb47b);
#             font-family: 'Arial', sans-serif;
#         }
#         .reportview-container {
#             background-color: rgba(255, 255, 255, 0.9);
#             border-radius: 10px;
#             padding: 20px;
#             box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
#         }
#         h1 {
#             color: #800080;
#             text-align: center;
#         }
#         .stTextInput>div>div>input {
#             border-color: #ff6347;
#             border-radius: 5px;
#             padding: 10px;
#             font-size: 16px;
#         }
#         .stButton>button {
#             background-color: #ff6347;
#             color: white;
#             border-radius: 5px;
#             padding: 10px 20px;
#             font-size: 16px;
#             transition: background-color 0.3s;
#         }
#         .stButton>button:hover {
#             background-color: #ff4500;
#         }
#         .stExpander {
#             background-color: #fff3e0;
#             border-radius: 5px;
#             padding: 10px;
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
#             st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

#             # Get the current timestamp
#             timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

#             # Save the user input and chatbot response to the chat_log.csv file
#             with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
#                 csv_writer = csv.writer(csvfile)
#                 csv_writer.writerow([user_input_str, response, timestamp])

#             if response.lower() in ['goodbye', 'bye']:
#                 st .write("Thank you for chatting with me. Have a great day!")
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




# fouth
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

ssl._create_default_https_context = ssl._create_unverified_context
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

# training the model
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
        """
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
        """, 
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
            st.text_area("Chatbot :", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

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
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f":User  {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

        st.subheader("Project Overview:")

        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
        """)

        st.subheader("Dataset:")

        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
        - Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
        - Text: The user input text.
        """)

        st.subheader("Streamlit Chatbot Interface:")

        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("Conclusion:")

        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")

if __name__ == '__main__':
    main()
