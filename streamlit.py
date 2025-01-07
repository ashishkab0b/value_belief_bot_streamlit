# streamlit.py

import streamlit as st
import os
from datetime import datetime, timezone
import yaml
import random
from bot import BotStart, BotIssue, BotRateIssue, BotGenerateReaps, BotRateReap, BotEnd
import pathlib
from bs4 import BeautifulSoup
from logger_setup import setup_logger
from config import CurrentConfig
from crud import (
    db_create_participant,
    db_get_participant,
)
from db import get_session
from models import (
    StateEnum,
    DomainEnum,
    RoleEnum
)
from utils import disable_copy_paste
from chat_session import load_chat

# Set up logger
logger = setup_logger()
logger.setLevel(CurrentConfig.LOG_LEVEL)

# Initialize page
st.set_page_config(page_title="Chatbot", page_icon="📖", layout="wide")
st.title("Interviewer Chatbot")

# Disable copy-paste
disable_copy_paste(st)

# Load bot messages
with open("bot_msgs.yml", "r") as f:
    bot_msgs = yaml.safe_load(f)


# Define the mapping of bot states to bot classes
bot_steps = {
    StateEnum.start: BotStart,
    StateEnum.issue: BotIssue,
    StateEnum.rate_issue: BotRateIssue,
    StateEnum.generate_reaps: BotGenerateReaps,
    StateEnum.rate_reap: BotRateReap,
    StateEnum.end: BotEnd
}

# Check for new session flag in URL and set default if missing
# if "new_session" not in st.query_params or st.query_params["new_session"] not in ["0", "1"]:
#     st.query_params["new_session"] = False
# else:
#     st.query_params["new_session"] = True if st.query_params["new_session"] == "1" else False

# Check for Prolific ID in URL and throw error if missing
if "prolific_id" not in st.query_params:
    st.error("Please provide a valid URL that includes the Prolific ID. Contact the researcher on Prolific for assistance.")
    st.stop()
    
# Set the session state prolific_id
if st.query_params["prolific_id"] == "test":
    # if prolific_id is "test", create a new unique test ID
    st.session_state["prolific_id"] = "test-" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    st.query_params["prolific_id"] = st.session_state["prolific_id"]
else:
    st.session_state["prolific_id"] = st.query_params["prolific_id"]

# if there is not an active streamlit session running, load the chat from the database or create a new chat
if not st.session_state.get("active_session", False): 
    logger.info(f"New chat session started for participant.id={st.session_state['prolific_id']}")
    with get_session() as session:
        # Load the participant from the database or create a new participant
        participant = db_get_participant(session, st.session_state["prolific_id"])
        if participant is None:
            logger.info(f"No participant found -- Creating new participant for participant.id={st.session_state['prolific_id']}")
            cur_domain = DomainEnum.career if random.random() < 0.5 else DomainEnum.relationship
            participant = db_create_participant(session=session,
                                                prolific_id=st.session_state["prolific_id"],
                                                cur_domain=cur_domain)
            session.commit()
        
        # Load the chat session from the database or initialize a new chat session and set it to the session state
        ss = load_chat(session=session, participant=participant)
        st.session_state.update(ss)
        session.commit()
        
# Log state
logger.info(f"Current chat state: {st.session_state['cur_state']}, participant.id={st.session_state['prolific_id']}")

# Get user input
user_input = st.chat_input()
logger.info(f"User input: {user_input}, participant.id={st.session_state['prolific_id']}, state={st.session_state['cur_state']}, domain={st.session_state['cur_domain']}")

# Display the session state for debugging
if CurrentConfig.DEBUG:
    view_state = st.expander("View the session state")
    with view_state:
        st.write(st.session_state)
        


# Init the correct bot step
bot = bot_steps[st.session_state["cur_state"]](st)

# Write existing messages to the chat  
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Add user input to the message list 
if user_input:

    # Write the user input to the chat
    with st.chat_message('user'):
        st.write(user_input)

    # Process the user input
    bot.process_input(user_input)

# Update the state
old_state = st.session_state["cur_state"]
new_state, next_output_kwargs = bot.next_state()
logger.info(f"Next chat state: {new_state}, cur_domain={st.session_state['cur_domain']}, participant.id={st.session_state['prolific_id']}")


    
# Init a new bot if needed for output
if old_state != new_state:
    bot = bot_steps[new_state](st)
    
# Generate the bot output
bot.generate_output(**next_output_kwargs) # optional: for when next_state provides data for the output


