# bot.py

from logger_setup import setup_logger
import yaml
from typing import Literal, Dict, Any, List, Tuple, Optional
import time
import json
import asyncio
import random
from decimal import Decimal
from datetime import datetime
from itertools import product
from openai import OpenAI
import openai
from outputs import (
    TextOutput, 
    RadioOutput, 
    RankReapsOutput,
    SliderOutput
)
from pprint import pprint
from crud import (
    db_get_participant,
    db_create_reappraisal,
    db_create_message,
    db_update_issue,
    db_update_reappraisal,
    db_get_issue_by_participant_and_domain,
    db_get_reappraisals_by_participant_and_domain
)
from sqlalchemy.orm import Session
from config import CurrentConfig
import streamlit as st
import os
from langsmith.wrappers import wrap_openai
from langsmith import traceable
import toml
from models import (
    Issue,
    Reappraisal, 
    Message,
    Participant,
    StateEnum,
    DomainEnum,
    RoleEnum,
    )

logger = setup_logger()
session = Session()

# load secrets toml into environment variables
secrets = toml.load(".streamlit/secrets.toml")
for key, value in secrets.items():
    os.environ[key] = str(value)
    

from sqlalchemy.orm import Session

from db import get_session
        
with open("bot_msgs.yml", "r") as f:
    bot_msgs = yaml.safe_load(f)

openai_api_key = CurrentConfig.OPENAI_API_KEY


N_REAPPRAISALS = CurrentConfig.N_REAPPRAISALS
    
with open("prompts.yml", "r") as f:
    prompts = yaml.safe_load(f)
    

# Auto-trace LLM calls in-context
client = wrap_openai(openai.Client(api_key=openai_api_key))
os.environ["LANGCHAIN_TRACING_V2"] = CurrentConfig.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = CurrentConfig.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = CurrentConfig.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = CurrentConfig.LANGCHAIN_PROJECT

class Chatbot():
    
    @staticmethod
    @traceable
    def query_gpt(prompt: str, messages: Optional[List[dict]] = None, max_tries: int = 3, pid: str=None) -> str:
        """
        A function that queries the GPT model.
        
        Parameters:
            prompt (str): The initial system prompt.
            messages (Optional[List[dict]]): Additional messages to include in the query.
            max_tries (int): Maximum number of retries in case of failure.
            pid (str): The prolific ID of the participant. (used for logging)
        
        Returns:
            str: The model's response or an error message.
        """
        model = CurrentConfig.LLM_MODEL
        temperature = CurrentConfig.LLM_TEMPERATURE
        msgs = [{"role": "system", "content": prompt}] + (messages or [])
        
        for attempt in range(max_tries):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    temperature=temperature,
                )
                return completion.choices[0].message.content
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e} -- pid={pid}")
        
        # If all attempts fail
        logger.error("Max retries reached. Returning fallback response -- pid={pid}")
        return CurrentConfig.ERROR_MESSAGE
    
class BotStep():

    def __init__(self, st):
        
        self.st = st  # streamlit object
        self.ss = st.session_state
        self.messages = st.session_state["messages"]
        self.user_input = None
        
    
    def process_input(self, user_input: str):
        """
        A function that saves the user's chat input to the database and session state.

        Args:
            user_input (str): The user's chat input
        """
        user_input = str(user_input).strip()

        try:
            with get_session() as session:
                message = db_create_message(
                    session=session,
                    participant_id=self.ss["prolific_id"],
                    role="user",
                    content=user_input,
                    state=self.ss["cur_state"], 
                    domain=self.ss["cur_domain"]
                )
                session.commit()
                message_id = message.id
        except Exception as e:
            logger.error(f"Error creating message -- participant_id={self.ss['prolific_id']}")
            logger.exception(e)
            self.st.error(CurrentConfig.ERROR_MESSAGE)
            self.st.stop()
        else:
            logger.info(f"Message created -- participant_id={self.ss['prolific_id']}, message_id={message_id}")
            self.ss['messages'].append({"role": "user", "content": user_input, "state": self.ss["cur_state"], "domain": self.ss["cur_domain"]})
            self.user_input = user_input
        
    def switch_cur_domain(self):
        '''
        A helper function that switches the current domain
        '''
        logger.info(f"Switching current domain from {self.ss['cur_domain']} -- participant_id={self.ss['prolific_id']}")
        if self.ss["cur_domain"] == DomainEnum.career:
            self.ss["cur_domain"] = DomainEnum.relationship
            self.ss["other_domain"] = DomainEnum.career
            with get_session() as session:
                participant = db_get_participant(session, self.ss["prolific_id"])
                participant.cur_domain = DomainEnum.relationship
                session.commit()
        elif self.ss["cur_domain"] == DomainEnum.relationship:
            self.ss["cur_domain"] = DomainEnum.career
            self.ss["other_domain"] = DomainEnum.relationship
            with get_session() as session:
                participant = db_get_participant(session, self.ss["prolific_id"])
                participant.cur_domain = DomainEnum.career
                session.commit()
        else:
            raise ValueError("Current domain is not set")
        
    def set_state(self, state: Literal[StateEnum.start, StateEnum.issue, StateEnum.rate_issue, StateEnum.generate_reaps, StateEnum.rate_reap, StateEnum.end]):
        '''
        A function that sets the current state
        '''
        self.ss["cur_state"] = state
        with get_session() as session:
            participant = db_get_participant(session, self.ss["prolific_id"])
            participant.cur_state = state
            session.commit()
        
    def domain_issue_interview_complete(self, domain: Literal[DomainEnum.career, DomainEnum.relationship]):
        '''
        A function that checks if all issues questions have been covered for a given domain
        '''
        msgs = self.ss["issue_messages"][domain]
        domain_finished = len(msgs) > 0 and "::finished::" in msgs[-1]["content"].lower()
        return domain_finished 
    
    def domain_reappraisal_generation_complete(self, domain: Literal[DomainEnum.career, DomainEnum.relationship]):
        '''
        A function that checks if all reappraisals have been generated for a given domain
        '''
        return len(self.ss["reappraisals"][domain]) == N_REAPPRAISALS
        
    def next_state(self) -> Tuple[str, str]:
        """
        A function that returns the next state and optional output.
        Since the same logic that identifies the next state is sometimes used to generate the output, this function allows you to pass output to the next stage.

        Raises:
            NotImplementedError: This function must be implemented in the subclass.

        Returns:
            Tuple[str, str]: Returns a tuple with the next state in index 0 and optionally data for the next output in index 1 (e.g., {"output": "some output"})
        """
        raise NotImplementedError
    
    
    def generate_output(self, **kwargs) -> List[Dict[str, Any]]:
        '''
        A function that generates the bot messages
        '''
        raise NotImplementedError
    
class BotStart(BotStep):
    
    def __init__(self, st):
        super().__init__(st)
        super().set_state(StateEnum.start)
        
    def process_input(self, user_input):
        if user_input:
            super().set_state(StateEnum.issue)
            bot = BotIssue(self.st)
            bot.process_input(user_input)
    
    def next_state(self) -> Tuple[str, str]:
        
        logger.debug(f"Entering BotStart.next_state -- participant_id={self.ss['prolific_id']}")
        
        # check if other domain is done
        other_domain_finished = self.domain_issue_interview_complete(self.ss["other_domain"])
        
        output = bot_msgs["issue_issue_transition"] + '\n\n' if other_domain_finished else ""
        # add solicit message for current domain
        output += bot_msgs["solicit_issue"][self.ss["cur_domain"]]
        
        super().set_state(StateEnum.issue)
        
        return StateEnum.issue, {"output": output}
    
    def generate_output(self, **kwargs):
        pass
    
    
class BotIssue(BotStep):
    
    
    def __init__(self, st):
        
        super().__init__(st)
        super().set_state(StateEnum.issue)
        self.gpt_output = None  # since we're generating gpt output to get next state anyway, we'll save as an attribute to avoid calling gpt twice
        
    def process_input(self, user_input):
        '''
        Save user input to database and to session state
        '''
        super().process_input(user_input)
        self.ss["issue_messages"][self.ss["cur_domain"]].append({
            "role": "user", 
            "content": user_input,
            "state": self.ss["cur_state"],
            "domain": self.ss["cur_domain"]
            })
        
    
    def next_state(self) -> Tuple[str, str]:
        '''
        A function that returns the next state.
        Possible cases
        - interview is ongoing
        - interview is finished, move to next interview
        - interview is finished, move to survey
        '''
        if not self.user_input:
            super().set_state(StateEnum.issue)
            return StateEnum.issue, {}
        
        # Query GPT for next issue question
        prompt = prompts["issue_interview"].format(domain=self.ss["cur_domain"])
        gpt_output = Chatbot.query_gpt(
            prompt=prompt, 
            messages=self.ss["issue_messages"][self.ss["cur_domain"]],
            pid=self.ss["prolific_id"]  # for logging
            )
        
        # Check if current interview is complete
        domain_is_finished = "::finished::" in gpt_output
        
        # In this case, to avoid having to call gpt twice, we'll save the output as an attribute to be used in generate_output()
        gpt_output = gpt_output.replace("::finished::", "")
        self.gpt_output = gpt_output
        
        # FOR TESTING
        # domain_is_finished = True
        
        if domain_is_finished:
            # add finished signal to issue_messages
            self.ss["issue_messages"][self.ss["cur_domain"]].append({
                "role": "assistant", 
                "content": "::finished::",
                "state": self.ss["cur_state"],
                "domain": self.ss["cur_domain"]
                })
            super().set_state(StateEnum.rate_issue)
            return StateEnum.rate_issue, {}
        else:
            super().set_state(StateEnum.issue)
            return StateEnum.issue, {"output": gpt_output}
    
    
    def generate_output(self, **kwargs):
        '''
        Generate interview questions regarding their issue
        '''
        output = kwargs.get("output")
        if output:
            msg_obj = {"role": "assistant", 
                       "content": output, 
                       "state": self.ss["cur_state"], 
                       "domain": self.ss["cur_domain"]}
            self.ss["issue_messages"][self.ss["cur_domain"]].append(msg_obj)
            self.ss["messages"].append(msg_obj)
            with self.st.chat_message("assistant"):
                self.st.markdown(output, unsafe_allow_html=True)
              
            try:  
                with get_session() as session:
                    message=db_create_message(
                        session=session,
                        participant_id=self.ss['prolific_id'],
                        role="assistant",
                        content=output,
                        state=self.ss["cur_state"],
                        domain=self.ss["cur_domain"]
                    )
                    session.commit()
                    message_id = message.id
            except Exception as e:
                logger.error(f"Error creating message -- participant_id={self.ss['prolific_id']}")
                logger.exception(e)
                self.st.error(CurrentConfig.ERROR_MESSAGE)
                self.st.stop()
            else:
                logger.info(f"Message created -- participant_id={self.ss['prolific_id']}, message_id={message_id}")
                
                
    
class BotRateIssue(BotStep):
        
        def __init__(self, st):
            super().__init__(st)
            super().set_state(StateEnum.rate_issue)
            
        def is_domain_finished(self, domain: Literal[DomainEnum.career, DomainEnum.relationship]):
            '''
            A function that checks if all issues have been rated for a given domain
            '''
            for q_label in bot_msgs[StateEnum.rate_issue].keys():
                if q_label not in self.ss['issues'][domain].keys() or self.ss['issues'][domain][q_label] is None:
                    return False
            return True
                
    
        def process_input(self, user_input):
            if user_input:
                super().process_input(user_input)
            
            
        def next_state(self) -> str:
            
            # check if the current domain is finished
            cur_dom_finished = self.is_domain_finished(self.ss["cur_domain"])
            if not cur_dom_finished:
                super().set_state(StateEnum.rate_issue)
                return StateEnum.rate_issue, {}
            
            other_dom_finished = self.domain_issue_interview_complete(self.ss["other_domain"])
            if other_dom_finished:
                # If other domain is finished, move to generating reappraisals
                if random.random() < 0.5: # randomize order of domains
                    self.switch_cur_domain()
                super().set_state(StateEnum.generate_reaps)
                return StateEnum.generate_reaps, {}
            else:
                # If other domain is not finished, move to next issue interview
                self.switch_cur_domain()
        
                # add solicit message for current domain
                output = bot_msgs["issue_issue_transition"]
                output += bot_msgs["solicit_issue"][self.ss["cur_domain"]]
                
                super().set_state(StateEnum.issue)
                
                return StateEnum.issue, {"output": output}

                
        def slider_callback(self):
            
            cur_dom = self.ss["cur_domain"]
            
            kw = {}
            for q_label, q_text in bot_msgs[StateEnum.rate_issue].items():
                q_key = q_label + "_" + cur_dom  # this is a label that gets assigned to the slider in the streamlit form
                self.ss["issues"][cur_dom][q_label] = self.st.session_state[q_key]
                if q_key in self.st.session_state:
                    kw[q_label] = self.st.session_state[q_key] if q_key in self.st.session_state else None
        
            try:
                with get_session() as session:
                    issue = db_get_issue_by_participant_and_domain(session, self.ss["prolific_id"], cur_dom)
                    db_update_issue(
                        session=session,
                        id=issue.id,
                        **kw
                    )
                    session.commit()
            except Exception as e:
                logger.error(f"Error updating issue with kw={kw}-- participant_id={self.ss['prolific_id']}")
                logger.exception(e)
                self.st.error(CurrentConfig.ERROR_MESSAGE)
                self.st.stop()
            else:
                logger.info(f"Issue updated: {kw} -- participant_id={self.ss['prolific_id']}, issue_type={cur_dom}")
                
                
        def generate_output(self, **kwargs):
            
            cur_dom = self.ss["cur_domain"]
            
            with st.form(key=StateEnum.rate_issue):
                for q_label, q_text in bot_msgs[StateEnum.rate_issue].items():
                    q_key = q_label + "_" + cur_dom
                    self.st.slider(q_text, min_value=0, max_value=100, step=1, key=q_key)
                self.st.form_submit_button("Continue", on_click=self.slider_callback)
                

class BotGenerateReaps(BotStep):
    
    '''Generate reappraisals'''
    
    def __init__(self, st):
        super().__init__(st)
        super().set_state(StateEnum.generate_reaps)
    
    def generate_reaps(self):
        issue_messages = self.ss["issue_messages"][self.ss["cur_domain"]]
        issue_messages = [{"role": msg["role"], "content": msg["content"]} for msg in issue_messages]
        prompt = prompts["n_reappraisals"].format(n=CurrentConfig.N_REAPPRAISALS)
        
        reap_generation_tries_remaining = 3
        while reap_generation_tries_remaining > 0:
            try:
                reap_resp = Chatbot.query_gpt(
                    prompt=prompt, 
                    messages=issue_messages,
                    pid=self.ss["prolific_id"]  # for logging
                )
                reap_texts = json.loads(reap_resp)
            except json.JSONDecodeError as e:
                reap_generation_tries_remaining -= 1
                logger.error(f"Error decoding JSON for response: {reap_resp} -- participant_id={self.ss['prolific_id']}. Attempts left: {reap_generation_tries_remaining}")
                logger.exception(e)
                if reap_generation_tries_remaining == 0:
                    self.st.error(CurrentConfig.ERROR_MESSAGE)
                    self.st.stop()
            else:
                logger.info(f"Reappraisals generated -- participant_id={self.ss['prolific_id']}")
                break
        
        reap_objs = []
        # Save to database
        with get_session() as session:
            # Get issue id
            issue = db_get_issue_by_participant_and_domain(session, self.ss["prolific_id"], self.ss["cur_domain"])
            # Save reappraisals to database
            for i, reap_text in enumerate(reap_texts):
                db_create_reappraisal(
                    session=session,
                    participant_id=self.ss["prolific_id"],
                    issue_id=issue.id,
                    domain=self.ss["cur_domain"],
                    reap_num=i,
                    text=reap_text
                )
                reap_objs.append({
                    "text": reap_text,
                    "reap_num": i
                })
            session.commit()
        
        return reap_objs
        
    def process_input(self, user_input):
        pass
        
    def next_state(self) -> Tuple[str, str]:
        super().set_state(StateEnum.rate_reap)
        return StateEnum.rate_reap, {"next_q_label": list(bot_msgs[StateEnum.rate_reap].keys())[0]}
            
    def generate_output(self, **kwargs):
        # Generate reappraisals
        reap_objs = self.generate_reaps()
        reap_texts = [reap["text"] for reap in reap_objs]
        
        # Save to session state
        self.ss["reappraisals"][self.ss["cur_domain"]] = reap_objs
        
        msg = "<p>"
        domain_txt = "career" if self.ss["cur_domain"] == DomainEnum.career else "relationship"
        msg += bot_msgs["reap_intro"].format(domain=domain_txt)
        msg += "</p>"
        
        # Retrieve summary of situation
        summary = BotSummarizeIssue(self.st).generate_summary(self.ss["issue_messages"][self.ss["cur_domain"]])
        # Reset state back correctly
        super().set_state(StateEnum.rate_reap)
        
        msg += "<p>"
        msg += '<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px;">'
        msg += "<strong>Issue summary: </strong>"
        msg += summary
        msg += "</div>"
        msg += "</p>"
    
        
        # Write instructions to the chat
        with self.st.chat_message("assistant"):
            self.st.markdown(msg, unsafe_allow_html=True)
        
        # Wait 3 seconds
        time.sleep(3)
          
        # Write reappraisals to the chat
        with self.st.form(key=StateEnum.generate_reaps):
            self.st.markdown("<h2>Perspectives:</h2>", unsafe_allow_html=True)
            cols = self.st.columns(CurrentConfig.N_REAPPRAISALS)
            for i, col in enumerate(cols):
                with col:
                    self.st.markdown(reap_texts[i], unsafe_allow_html=True)
            self.st.form_submit_button("Continue", on_click=lambda: True)
        

class BotRateReap(BotStep):
    
    def __init__(self, st):
        super().__init__(st)
        super().set_state(StateEnum.rate_reap)
    
    def q_completions(self, domain: Literal[DomainEnum.career, DomainEnum.relationship]) -> Dict[str, list]:
        '''
        A function that returns the remaining reappraisal questions
        
        Returns: a dictionary with the question label as key and the reappraisal nums for whom the question is not yet completed as a list of values
        '''
        
        remaining = {q_label: [] for q_label in bot_msgs[StateEnum.rate_reap].keys()}
        
        # check if all reappraisals exist for this domain
        if len(self.ss["reappraisals"][domain]) < N_REAPPRAISALS:
            # if not, return all questions as remaining
            remaining = {q_label: list(range(N_REAPPRAISALS)) for q_label in bot_msgs[StateEnum.rate_reap].keys()}
            logger.debug(f"No reaps for domain so all questions remaining for domain={domain}: {remaining} -- participant_id={self.ss['prolific_id']}")
            return remaining
        
        # check if which questions are remaining to be answered
        for q_label in bot_msgs[StateEnum.rate_reap].keys():
            for i in range(N_REAPPRAISALS):
                if self.ss["reappraisals"][domain][i].get(q_label, None) is None:
                    remaining[q_label].append(i)
                    break
                    
                # slider_key = q_label + "_" + domain + "_" + str(i)
                # if slider_key not in self.st.session_state:  # note this checks the slider_key in the session state, not the rating in the organized reappraisal object
                #     remaining[q_label].append(i)
        logger.debug(f"Remaining questions for domain={domain}: {remaining} -- participant_id={self.ss['prolific_id']}")            
        return remaining  # e.g. {"success": [4, 5], "valued": [2, 3]}
    
    def remaining_questions(self, domain: Literal[DomainEnum.career, DomainEnum.relationship]) -> List[str]:
        '''
        A function that returns the remaining question labels
        '''
        q_completions = self.q_completions(domain)
        remaining_qs = [k for k, v in q_completions.items() if v]
        return remaining_qs
            
    def process_input(self, user_input):
        super().process_input(user_input)
        
    def next_state(self) -> str:
        
        q_completions_cur = self.q_completions(self.ss["cur_domain"])
        remaining_qs_cur = [k for k, v in q_completions_cur.items() if v]  # get q labels for which there are remaining questions
        
        q_completions_other = self.q_completions(self.ss["other_domain"])
        remaining_qs_other = [k for k, v in q_completions_other.items() if v]
        
        if not remaining_qs_cur and not remaining_qs_other:
            super().set_state(StateEnum.end)
            return StateEnum.end, {}
        elif not remaining_qs_cur:
            self.switch_cur_domain()
            super().set_state(StateEnum.generate_reaps)
            return StateEnum.generate_reaps, {}
        elif remaining_qs_cur:
            super().set_state(StateEnum.rate_reap)
            return StateEnum.rate_reap, {"next_q_label": remaining_qs_cur[0]}
    
    def slider_callback(self, q_label):
        domain = self.ss["cur_domain"]
        for i in range(N_REAPPRAISALS):
            slider_key = q_label + "_" + domain + "_" + str(i)
            label, _, i = slider_key.split("_")
            i = int(i)
            self.ss["reappraisals"][domain][int(i)][label] = self.st.session_state[slider_key]
            
            try:
                with get_session() as session:
                    reaps = db_get_reappraisals_by_participant_and_domain(session, self.ss["prolific_id"], domain)
                    reap = [reap for reap in reaps if reap.reap_num == i][0]
                    kw = {label: self.st.session_state[slider_key]}
                    reap_id = reap.id
                    db_update_reappraisal(
                        session=session,
                        id=reap_id,
                        **kw
                    )
                    session.commit()
            except Exception as e:
                logger.error(f"Error updating reappraisal with kw={kw}-- participant_id={self.ss['prolific_id']}")
                logger.exception(e)
                self.st.error(CurrentConfig.ERROR_MESSAGE)
                self.st.stop()
            else:
                logger.info(f"Reappraisal updated -- participant_id={self.ss['prolific_id']}, reappraisal_id={reap_id}")
                
            
    def generate_output(self, **kwargs):
        '''
        Ask same question for all 5 reappraisals in one page
        '''
        
        if "next_q_label" in kwargs:
            next_q_label = kwargs["next_q_label"]
        else:
            remaining_qs = self.remaining_questions(self.ss["cur_domain"])
            next_q_label = remaining_qs[0]
        
        # Loading next question message (to highlight that the question is changing)
        with self.st.chat_message("assistant"):
            self.st.markdown(bot_msgs["rate_reap_next_q"], unsafe_allow_html=True)
        time.sleep(2)
        
        with self.st.form(key=StateEnum.rate_reap):
            
            # Print the question
            with self.st.chat_message("assistant"):
                self.st.markdown(bot_msgs["rate_reap_stem"], unsafe_allow_html=True)
                # self.st.markdown(bot_msgs[StateEnum.rate_reap][next_q_label], unsafe_allow_html=True)
            
            q_col, reap_col = self.st.columns([1, 3])
            with q_col:
                self.st.markdown(bot_msgs[StateEnum.rate_reap][next_q_label], unsafe_allow_html=True)
            with reap_col:
                # Loop through each reappraisal and ask the user to rate it
                for i, reap in enumerate(self.ss["reappraisals"][self.ss["cur_domain"]]):
                    reap_text = reap["text"]
                    q_key = next_q_label + "_" + self.ss["cur_domain"] + "_" + str(i)
                    self.st.slider(reap_text, min_value=0, max_value=100, step=1, key=q_key)
            
            self.st.form_submit_button("Continue", on_click=lambda: self.slider_callback(next_q_label))

        return
    

class BotSummarizeIssue(BotStep):
    
    def __init__(self, st):
        super().__init__(st)
        super().set_state(StateEnum.summarize_issue)
        
    def generate_summary(self, issue_messages: List[dict]):
        prompt = prompts["summarize_issue"]
        msgs = [{"role": msg["role"], "content": msg["content"]} for msg in issue_messages]
        summary = Chatbot.query_gpt(
            prompt=prompt, 
            messages=msgs,
            pid=self.ss["prolific_id"]  # for logging
            )
        
        try:
            with get_session() as session:
                issue = db_get_issue_by_participant_and_domain(session, self.ss["prolific_id"], self.ss["cur_domain"])
                db_update_issue(
                    session=session,
                    id=issue.id,
                    summary=summary
                )
                session.commit()
        except Exception as e:
            logger.error(f"Error updating issue with summary={summary}-- participant_id={self.ss['prolific_id']}")
            logger.exception(e)
            self.st.error(CurrentConfig.ERROR_MESSAGE)
            self.st.stop()
        else:
            logger.info(f"Issue updated -- participant_id={self.ss['prolific_id']}, issue_type={self.ss['cur_domain']}")
            logger.debug(f"Updated issue: {summary}")
            
        return summary      
    
    def process_input(self, user_input):
        return
        
    def next_state(self) -> str:
        return None, None
            
    def generate_output(self, **kwargs):
        return 
    

# class BotSurvey(BotStep):
    
#     def __init__(self, st):
#         super().__init__(st)
#         self.ss['cur_state'] = "survey"
    
#     def process_input(self, user_input):
#         return
        
#     def next_state(self) -> str:
#         return None, None
            
#     def generate_output(self, **kwargs):
#         return 

class BotEnd(BotStep):
    
    def __init__(self, st):
        super().__init__(st)
        super().set_state(StateEnum.end)
        
    def process_input(self, user_input):
        super().process_input(user_input=user_input)
        return
        
    def next_state(self) -> str:
        return None, None
            
    def generate_output(self, **kwargs):
        
        output = bot_msgs["end_chatbot"]
        with self.st.chat_message("assistant"):
            self.st.markdown(output, unsafe_allow_html=True)
      
        return 
