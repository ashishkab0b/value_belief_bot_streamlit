import streamlit as st
import os
from datetime import datetime
import yaml
import random
from bot import BotStart, BotIssue, BotRateIssue, BotGenerateReaps, BotRateReap, BotEnd
from db import (
    db_new_participant
)
from config import Config
import pathlib
from bs4 import BeautifulSoup

# Disable copy/paste
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

# Insert the script in the head tag of the static template inside your virtual environement
index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
soup = BeautifulSoup(index_path.read_text(), features="lxml")
if not soup.find(id='custom-js'):
    script_tag = soup.new_tag("script", id='custom-js')
    script_tag.string = GA_JS
    soup.head.append(script_tag)
    index_path.write_text(str(soup))
    
    
    
st.set_page_config(page_title="Chatbot", page_icon="📖")
st.title("Interviewer Chatbot")

    
with open("bot_msgs.yml", "r") as f:
    bot_msgs = yaml.safe_load(f)

config = Config()   

# Initialize the session state
init_state = {
    "messages": [],  # (role, content)
    "state": "start",
    "llm_model": config.llm_model,
    "pid": st.query_params["pid"] if "pid" in st.query_params else "00000",
    "started_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    # "issue_career_msgs": [],
    # "issue_relationship_msgs": [],
    "issue_msgs": {"career": [], "relationship": []},
    "rate_issue": {"career": {}, "relationship": {}}, # {career: {pos: 3, pos_changed: True, }}
    "reaps": {"career": [], "relationship": []},
    # "reap_ranks": {"career": [], "relationship": []},
    "rate_reap": {"career": [], "relationship": []},   # {career: [{success: 3, success_changed: True, }, {}]}
    "cur_domain": "career" if random.random() < 0.5 else "relationship",
    "survey_order": random.sample(["beliefs", "values"], 2),
    "cur_survey_idx": 0,
    "show_chat": True,
    }
for key, value in init_state.items():
    if key not in st.session_state:
        st.session_state[key] = value
        
if st.session_state["state"] == "start":
    db_new_participant(st.session_state["pid"])
    # Add first bot message
    # first_msg = bot_msgs["solicit_issue"][st.session_state["cur_domain"]]
    # st.session_state["messages"].append({"role": "assistant", "content": first_msg})
    # st.session_state["state"] = "issue"


bot_steps = {
    "start": BotStart,
    "issue": BotIssue,
    "rate_issue": BotRateIssue,
    # "survey": BotSurvey,
    "generate_reaps": BotGenerateReaps,
    "rate_reap": BotRateReap,
    "end": BotEnd
}
        

# if st.session_state["show_chat"]:
if True: # TODO: maybe will implement the real thing later - hide chat during questions
    user_input = st.chat_input()
else:
    user_input = None


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
# previous_show_chat_state = st.session_state["show_chat"]
new_state, next_output_kwargs = bot.next_state()
# if previous_show_chat_state != st.session_state["show_chat"]:
#     st.rerun()
    
# Init a new bot if needed for output
if old_state != new_state:
    bot = bot_steps[new_state](st)
    
# Generate the bot output
bot.generate_output(**next_output_kwargs) # optional: for when next_state provides data for the output

# Display the session state for debugging
# view_state = st.expander("View the session state")
# with view_state:
#     st.write(st.session_state)
    

