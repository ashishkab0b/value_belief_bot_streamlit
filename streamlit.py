# streamlit.py

import streamlit as st
import os
from datetime import datetime
import yaml
import random
from bot import BotStart, BotIssue, BotRateIssue, BotGenerateReaps, BotRateReap, BotEnd
import pathlib
from bs4 import BeautifulSoup
from logger_setup import setup_logger
from config import CurrentConfig
from crud import (
    db_create_participant,
    db_create_issue,
    db_create_message,
    db_create_reappraisal,
    db_get_issue_by_id,
    db_get_reappraisal_by_id,
    db_update_issue,
    db_update_reappraisal,
)
from db import get_session

# Set up logger
logger = setup_logger()
logger.setLevel(CurrentConfig.LOG_LEVEL)


# Disable copy/paste by inserting javascript into default streamlit index.html
GA_JS = """
document.addEventListener('DOMContentLoaded', function() {
    // Disable text selection
    document.body.style.userSelect = 'none';

    // Disable copy-paste events
    document.addEventListener('copy', (e) => {
        e.preventDefault();
    });
    document.addEventListener('paste', (e) => {
        e.preventDefault();
    });
});
"""
index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
soup = BeautifulSoup(index_path.read_text(), features="lxml")
if not soup.find(id='custom-js'):
    script_tag = soup.new_tag("script", id='custom-js')
    script_tag.string = GA_JS
    soup.head.append(script_tag)
    index_path.write_text(str(soup))
    
    
    
# Initialize page
st.set_page_config(page_title="Chatbot", page_icon="📖")
st.title("Interviewer Chatbot")


# Load bot messages
with open("bot_msgs.yml", "r") as f:
    bot_msgs = yaml.safe_load(f)


# Define the mapping of bot states to bot classes
bot_steps = {
    "start": BotStart,
    "issue": BotIssue,
    "rate_issue": BotRateIssue,
    "generate_reaps": BotGenerateReaps,
    "rate_reap": BotRateReap,
    "end": BotEnd
}

# Initialize the session state
init_state = {
    "messages": [],  # (role, content)
    "state": "start",
    "llm_model": CurrentConfig.LLM_MODEL,
    "prolific_id": st.query_params["prolific_id"] if "prolific_id" in st.query_params else "00000",
    "started_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "issue_msgs": {"career": [], "relationship": []},
    "rate_issue": {"career": {}, "relationship": {}}, # {career: {pos: 3, pos_changed: True, }}
    "reaps": {"career": [], "relationship": []},
    "rate_reap": {"career": [], "relationship": []},   # {career: [{success: 3, success_changed: True, }, {}]}
    "cur_domain": "career" if random.random() < 0.5 else "relationship",
    "survey_order": random.sample(["beliefs", "values"], 2),
    # "cur_survey_idx": 0,
    "show_chat": True,  # TODO
    }
for key, value in init_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Initialize starting data in the database
if st.session_state["state"] == "start":
    # Initialize the participant in the database
    try:
        with get_session() as session:
            # Create the participant
            logger.info(f"Initializing participant -- prolific_id={st.session_state['prolific_id']}")
            participant = db_create_participant(session, st.session_state["prolific_id"])
            session.flush()
            st.session_state["pid"] = participant.id
            session.commit()
            
    except Exception as e:
        logger.error(f"Error initializing participant -- prolific_id={st.session_state['prolific_id']}")
        logger.exception(e)
        st.error(CurrentConfig.ERROR_MESSAGE)
        st.stop()
    else:
        logger.info(f"Participant initialized -- id={st.session_state['pid']}, prolific_id={st.session_state['prolific_id']}")
    
    # Initialize issues and reappraisals in the database
    try:
        with get_session() as session:
            
            # Create two issues
            issue_career = db_create_issue(session, st.session_state["pid"], domain="career")
            issue_relationship = db_create_issue(session, st.session_state["pid"], domain="relationship")
            session.flush()
            st.session_state["issue_ids"] = {
                "career": issue_career.id,
                "relationship": issue_relationship.id
            }
            session.commit()
    except Exception as e:
        logger.error(f"Error initializing issues -- participant.id={st.session_state['pid']}")
        logger.exception(e)
        st.error(CurrentConfig.ERROR_MESSAGE)
        st.stop()
    else:
        logger.info(f"Issues initialized -- participant.id={st.session_state['pid']}")
        
    # Initialize reappraisals
    try:   
        # Create reappraisals
        st.session_state["reappraisal_ids"] = {
            "career": [],
            "relationship": []
        }
        with get_session() as session:
            for i in range(CurrentConfig.N_REAPPRAISALS):
                reappraisal_career = db_create_reappraisal(session, 
                                                           participant_id=st.session_state["pid"],
                                                           domain="career", 
                                                           issue_id=st.session_state["issue_ids"]["career"],
                                                           reap_num=i+1)
                reappraisal_relationship = db_create_reappraisal(session, 
                                                                participant_id=st.session_state["pid"],
                                                                domain="relationship", 
                                                                issue_id=st.session_state["issue_ids"]["relationship"],
                                                                reap_num=i+1)
                session.flush()
                st.session_state["reappraisal_ids"]["career"].append(reappraisal_career.id)
                st.session_state["reappraisal_ids"]["relationship"].append(reappraisal_relationship.id)
            session.commit()
    except Exception as e:
        logger.error(f"Error initializing reappraisals -- participant.id={st.session_state['pid']}")
        logger.exception(e)
        st.error(CurrentConfig.ERROR_MESSAGE)
        st.stop()
    else:
        logger.info(f"Reappraisals initialized -- participant.id={st.session_state['pid']}")
        
        
        
    
# Log state
logger.info(f"Current chat state: {st.session_state['state']}, participant.id={st.session_state['pid']}")

# Get user input
user_input = st.chat_input()
logger.info(f"User input: {user_input}, participant.id={st.session_state['pid']}")


# Init the correct bot step
bot = bot_steps[st.session_state["state"]](st)

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
old_state = st.session_state["state"]
new_state, next_output_kwargs = bot.next_state()
logger.info(f"Next chat state: {new_state}, participant.id={st.session_state['pid']}")

    
# Init a new bot if needed for output
if old_state != new_state:
    bot = bot_steps[new_state](st)
    
# Generate the bot output
bot.generate_output(**next_output_kwargs) # optional: for when next_state provides data for the output

# Display the session state for debugging
if CurrentConfig.DEBUG:
    view_state = st.expander("View the session state")
    with view_state:
        st.write(st.session_state)
        

