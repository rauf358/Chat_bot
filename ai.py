import streamlit as st  # Import Streamlit for creating web applications
from openai import OpenAI  # Import OpenAI library
import os  # Import os module for interacting with the operating system
import tiktoken  # Import tiktoken for token encoding
from dotenv import load_dotenv  # Import load_dotenv to load environment variables
load_dotenv()  # Load environment variables from .env file

api_key = os.getenv('api_key')  # Get API key from environment variables

class CoversationalBot:
    def __init__(self,model ="gpt-3.5-turbo",base_url = "https://api.openai.com/v1.",token_budget = 500):
        # Initialize the ConversationalBot class with default parameters
        self.model = model  # Set the GPT model (default: gpt-3.5-turbo)
        self.base_url = base_url  # Set the base URL for OpenAI API
        self.client = OpenAI(api_key=api_key)  # Initialize OpenAI client with API key
        self.token_budget = token_budget  # Set the token budget
        
        # Define system messages for different behaviors
        self.sys_messages = {
            "Helpfull": "You are a helpful assistant.",
            "Sassy": "You are a sassy assistant that is fed up with answering questions.",
            "Angry": "You are an angry assistant that likes yelling in all caps.",
            "Funny": "You are a funny assistant who jokes a lot.",
            "Lonely": "You are a lonely assistant that gives depressed answers."
        }
        self.sys = self.sys_messages["Helpfull"]  # Set default system behavior to helpful
        self.convo_history = [{"role": "system", "content": self.sys}]  # Initialize conversation history with default system message
    
    # Function to calculate tokens used by a text
    def token_calculate(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.encoding_for_model('cl100k_base')
        tokens = encoding.encode(text)
        return len(tokens)
   
    # Function to calculate total tokens used in the conversation history
    def total_token_used(self):
        token_used = sum(self.token_calculate(d["content"]) for d in self.convo_history)
        return token_used
    
    # Function to enforce token limit by removing older messages if exceeded
    def enforce_token_limit(self):
        while self.total_token_used() > self.token_budget:
            if len(self.convo_history) <= 1:
                break
            self.convo_history.pop(1)
   
    # Function to set system message based on behavior
    def system_message(self, key):
        if self.convo_history and self.convo_history[0]["role"] == 'system':
            self.convo_history[0]["content"] = self.sys_messages[key]
        else:
            self.convo_history.insert(0, {"role": "system", "content": self.sys_messages[key]})     
        
    # Function to generate AI response given user input
    def prompt(self, prompt, key, temp=0.2, maxt=200):
        self.system_message(key)  # Set system message based on behavior
        self.enforce_token_limit()  # Enforce token limit
        self.convo_history.append({"role": "user", "content": prompt})  # Append user input to conversation history
        # Generate AI response using OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.convo_history,
            temperature=temp,
            max_tokens=maxt
        )
        ai_response = response.choices[0].message.content  # Get AI response
        self.convo_history.append({"role": "assistant", "content": ai_response})  # Append AI response to conversation history
        return ai_response

# Set up Streamlit interface
st.image("chat bot logo.jpg")  # Display image
st.title(" Advisor AI CHAT BOT :robot_face:")  # Set title for the chatbot

# Initialize ConversationBot
bot = CoversationalBot()

# Initial message to the user
with st.chat_message('ai'):
    st.write("How can I help you?")

# User input field
user_input = st.chat_input("Please ask your query")

# Sidebar option to change assistant behavior
change = st.sidebar.radio("Want to set System Behavior", ['Yes', "No"])
if change == "Yes":
    behavior = st.sidebar.selectbox("Select Assistant Behavior", [None, "Helpfull", "Sassy", 'Angry', 'Funny', 'Lonely'])
    if behavior is not None:
        bot.system_message(behavior)

# Sidebar sliders for setting temperature and max tokens
temp_val = st.sidebar.slider("Set Temperature", 0.0, 0.5, 1.0)
max_tokens = st.sidebar.slider("Set Max Tokens", 0, 200, 500)    

if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []
# Retrieve conversation history from session state
bot.convo_history = st.session_state['conversation_history']

# Generate response if user input is not None
if user_input is not None:
    bot.prompt(user_input, behavior, temp_val, max_tokens)

# Display conversation history
for d in bot.convo_history:   
    with st.chat_message(d['role']):
        st.write(d['content'])

# Reset conversation history with a button
def reset():
    st.session_state['conversation_history'] = []
st.sidebar.button("New Chat", on_click=reset)  # Callback method
