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
from outputs import (
    TextOutput, 
    RadioOutput, 
    RankReapsOutput,
    SliderOutput
)
from pprint import pprint
from db import (
    db_add_message,
    db_issue_feature,
    db_reap_feature,
    db_new_reap
)
from config import Config
import streamlit as st

logger = setup_logger()

with open("bot_msgs.yml", "r") as f:
    bot_msgs = yaml.safe_load(f)

openai_api_key = st.secrets["OPENAI_API_KEY"]

config = Config()

N_REAPPRAISALS = config.n_reappraisals
    
with open("prompts.yml", "r") as f:
    prompts = yaml.safe_load(f)

class Chatbot():
    
    @staticmethod
    def query_gpt(prompt: str, messages: Optional[List[dict]] = None, max_tries: int = 3, pid=None) -> str:
        """
        A function that queries the GPT model.
        
        Parameters:
            prompt (str): The initial system prompt.
            messages (Optional[List[dict]]): Additional messages to include in the query.
            max_tries (int): Maximum number of retries in case of failure.
        
        Returns:
            str: The model's response or an error message.
        """
        client = OpenAI(api_key=openai_api_key)
        model = config.llm_model
        temperature = config.temperature
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
                logger.error(f"PID {pid} - Attempt {attempt + 1} failed: {e}")
        
        # If all attempts fail
        logger.error("PID {pid} - Max retries reached. Returning fallback response.")
        return config.error_message
    
class BotStep():

    def __init__(self, st):
        
        self.st = st  # streamlit object
        self.ss = st.session_state
        self.cur_domain = st.session_state["cur_domain"]
        self.other_domain = "career" if self.cur_domain == "relationship" else "relationship"
        self.pid = st.session_state["pid"]
        self.cur_state = st.session_state["state"]
        self.messages = st.session_state["messages"]
        self.user_input = None
        
    
    def process_input(self, user_input):
        # Save user input to db
        db_add_message(
            pid=self.pid, 
            role='user', 
            content=user_input,
            state=self.ss["state"],
            domain=self.ss["cur_domain"]
            )
        
        self.messages.append({"role": "user", "content": user_input})
        # Store user input as an attribute
        self.user_input = user_input
        
    def switch_cur_domain(self):
        '''
        A helper function that switches the current domain
        '''
        if self.ss["cur_domain"] == "career":
            self.ss["cur_domain"] = "relationship"
            self.cur_domain = "relationship"
            self.other_domain = "career"
        elif self.ss["cur_domain"] == "relationship":
            self.ss["cur_domain"] = "career"
            self.cur_domain = "career"
            self.other_domain = "relationship"
        else:
            raise ValueError("Current domain is not set")
        
    def domain_issue_interview_complete(self, domain: Literal["career", "relationship"]):
        '''
        A function that checks if all issues have been rated for a given domain
        '''
        domain_finished = len(self.ss["issue_msgs"][domain]) > 0 and "::finished::" in self.ss["issue_msgs"][domain][-1]["content"].lower()
        return domain_finished 
    
    def domain_reappraisal_generation_complete(self, domain: Literal["career", "relationship"]):
        '''
        A function that checks if all reappraisals have been generated for a given domain
        '''
        return len(self.ss["reaps"][domain]) == N_REAPPRAISALS
        
    def next_state(self):
        '''
        A function that returns the next state'''
        raise NotImplementedError
    
    
    def generate_output(self, **kwargs) -> List[Dict[str, Any]]:
        '''A function that generates the bot messages'''
        raise NotImplementedError
    
class BotStart(BotStep):
    
    def __init__(self, st):
        super().__init__(st)
        self.cur_state = "start"
        
    def process_input(self, user_input):
        if user_input:
            self.ss["state"] = "issue"
            bot = BotIssue(self.st)
            bot.process_input(user_input)
    
    def next_state(self) -> Tuple[str, str]:
        
        # check if other domain is done
        other_domain_finished = self.domain_issue_interview_complete(self.other_domain)
        
        output = bot_msgs["issue_issue_transition"] + '\n\n' if other_domain_finished else ""
        # add solicit message for current domain
        output += bot_msgs["solicit_issue"][self.ss["cur_domain"]]
        
        self.ss["state"] = "issue"
        
        return "issue", {"output": output}
    
    def generate_output(self, **kwargs):
        pass
    
    
class BotIssue(BotStep):
    
    
    def __init__(self, st):
        
        super().__init__(st)
        self.cur_state = "issue"
        self.gpt_output = None  # since we're generating gpt output to get next state anyway, we'll save as an attribute to avoid calling gpt twice
        
    def process_input(self, user_input):
        '''
        Save user input to database and to session state
        '''
        super().process_input(user_input)
        self.ss["issue_msgs"][self.ss["cur_domain"]].append({"role": "user", "content": user_input})
        
    
    def next_state(self) -> Tuple[str, str]:
        '''
        A function that returns the next state.
        Possible cases
        - interview is ongoing
        - interview is finished, move to next interview
        - interview is finished, move to survey
        '''
        if not self.user_input:
            self.ss["state"] = "issue"
            return "issue", {}
        
        # Query GPT for next issue question
        prompt = prompts["issue_interview"].format(domain=self.ss["cur_domain"])
        gpt_output = Chatbot.query_gpt(
            prompt=prompt, 
            messages=self.ss["issue_msgs"][self.ss["cur_domain"]],
            pid=self.ss["pid"]  # for logging
            )
        
        # Check if current interview is complete
        domain_is_finished = "::finished::" in gpt_output
        
        # In this case, to avoid having to call gpt twice, we'll save the output as an attribute to be used in generate_output()
        gpt_output = gpt_output.replace("::finished::", "")
        self.gpt_output = gpt_output
        
        # FOR TESTING
        # domain_is_finished = True
        
        if domain_is_finished:
            # add finished signal to issue_msgs
            self.ss["issue_msgs"][self.ss["cur_domain"]].append({"role": "assistant", "content": "::finished::"})
            self.ss['state'] = "rate_issue"
            self.ss["show_chat"] = False
            return "rate_issue", {}
        else:
            self.ss["state"] = "issue"
            self.ss["show_chat"] = True
            return "issue", {"output": gpt_output}
    
    
    def generate_output(self, **kwargs):
        '''
        Generate interview questions regarding their issue
        '''
        output = kwargs.get("output")
        if output:
            self.ss["issue_msgs"][self.ss["cur_domain"]].append({"role": "assistant", "content": output})
            self.ss["messages"].append({"role": "assistant", "content": output})
            with self.st.chat_message("assistant"):
                self.st.markdown(output, unsafe_allow_html=True)
        
            db_add_message(
                pid=self.pid, 
                role='assistant', 
                content=output,
                state=self.ss["state"],
                domain=self.ss["cur_domain"]
                )
                
    
class BotRateIssue(BotStep):
        
        def __init__(self, st):
            super().__init__(st)
            self.cur_state = "rate_issue"
            
        def is_domain_finished(self, domain: Literal["career", "relationship"]):
            '''
            A function that checks if all issues have been rated for a given domain
            '''
            for q_label in bot_msgs['rate_issue'].keys():
                if q_label not in self.ss['rate_issue'][domain].keys():
                    return False
            return True
                
    
        def process_input(self, user_input):
            if user_input:
                super().process_input(user_input)
            
            
        def next_state(self) -> str:
            
            other_dom_finished = self.domain_issue_interview_complete(self.other_domain)
            if other_dom_finished:
                # If other domain is finished, move to generating reappraisals
                if random.random() < 0.5: # randomize order of domains
                    self.switch_cur_domain()
                self.ss["state"] = "generate_reaps"
                self.ss["show_chat"] = True
                return "generate_reaps", {}
            else:
                # If other domain is not finished, move to next issue interview
                self.switch_cur_domain()
        
                # add solicit message for current domain
                output = bot_msgs["issue_issue_transition"]
                output += bot_msgs["solicit_issue"][self.ss["cur_domain"]]
                
                self.ss["state"] = "issue"
                self.ss["show_chat"] = True
                
                return "issue", {"output": output}

                
        def slider_callback(self):
            
            cur_dom = self.ss["cur_domain"]
            
            for q_label, q_text in bot_msgs["rate_issue"].items():
                q_key = q_label + "_" + cur_dom
                self.ss["rate_issue"][cur_dom][q_label] = self.st.session_state[q_key]
                db_issue_feature(pid=self.ss["pid"], 
                                 issue_type=cur_dom,
                                 feature_type=q_label.upper(),
                                 value=self.st.session_state[q_key])
            
                
        def generate_output(self, **kwargs):
            
            cur_dom = self.ss["cur_domain"]
            
            with st.form(key="rate_issue"):
                for q_label, q_text in bot_msgs["rate_issue"].items():
                    q_key = q_label + "_" + cur_dom
                    self.st.slider(q_text, min_value=0, max_value=100, step=1, key=q_key)
                self.st.form_submit_button("Continue", on_click=self.slider_callback)
                

class BotGenerateReaps(BotStep):
    
    '''Generate reappraisals'''
    
    def __init__(self, st):
        super().__init__(st)
        self.cur_state = "generate_reaps"
    
    def generate_reaps(self):
        issue_msgs = self.ss["issue_msgs"][self.ss["cur_domain"]]
        prompt = prompts["n_reappraisals"].format(n=config.n_reappraisals)
        reaps = Chatbot.query_gpt(
            prompt=prompt, 
            messages=issue_msgs,
            pid=self.ss["pid"]  # for logging
            )
        reaps = json.loads(reaps)
        return reaps
        
    def process_input(self, user_input):
        pass
        
    def next_state(self) -> Tuple[str, str]:
        self.ss["state"] = "rate_reap"
        self.ss["show_chat"] = False
        return "rate_reap", {"next_q_label": list(bot_msgs['rate_reap'].keys())[0]}
            
    def generate_output(self, **kwargs):
        # Generate reappraisals
        reaps = self.generate_reaps()
        
        # Save to object and session state
        self.reaps = reaps
        self.ss["reaps"][self.ss["cur_domain"]] = reaps
        
        # Initialize reappraisal rating dictionary
        for reap in reaps:
            reap_responses = {}
            for q_label in bot_msgs["rate_reap"].keys():
                reap_responses["reap"] = reap  # TODO: perhaps remove in prod to reduce data transfer and storage
                reap_responses[q_label] = None
            self.ss["rate_reap"][self.ss["cur_domain"]].append(reap_responses)
        
        msg = "<p>"
        msg += bot_msgs["reap_intro"].format(domain=self.ss["cur_domain"])
        msg += "</p>"
        
        # Retrieve summary of situation
        summary = BotSummarizeIssue(self.st).generate_summary(self.ss["issue_msgs"][self.ss["cur_domain"]])
        msg += "<p>"
        msg += '<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px;">'
        msg += "<strong>Issue summary: </strong>"
        msg += summary
        msg += "</div>"
        msg += "</p>"
        db_issue_feature(
            pid=self.ss["pid"],
            issue_type=self.ss["cur_domain"],
            feature_type="SUMMARY",
            value=summary
        )
        
        # Write instructions to the chat
        with self.st.chat_message("assistant"):
            self.st.markdown(msg, unsafe_allow_html=True)
        
        # Wait 3 seconds
        time.sleep(3)
          
        # Write reappraisals to the chat
        with self.st.form(key="generate_reaps"):
            self.st.markdown("<h2>Perspectives:</h2>", unsafe_allow_html=True)
            cols = self.st.columns(config.n_reappraisals)
            for i, col in enumerate(cols):
                with col:
                    self.st.markdown(reaps[i], unsafe_allow_html=True)
            self.st.form_submit_button("Continue", on_click=lambda: True)
        
        # Add reappraisals to database
        for i, reap in enumerate(reaps):
            db_new_reap(
                pid=self.ss["pid"], 
                issue_type=self.ss["cur_domain"], 
                reap_num=i, 
                reap_text=reap)
                

class BotRateReap(BotStep):
    
    def __init__(self, st):
        super().__init__(st)
        self.cur_state = "rate_reap"
    
    def q_completions(self, domain: Literal["career", "relationship"]):
        '''
        A function that returns the remaining reappraisal questions
        
        Returns: a dictionary with reappraisal index as key and lists of remaining question keys as value
        '''
        remaining = {}
        for q_label in bot_msgs['rate_reap'].keys():
            if len(self.ss['rate_reap'][domain]) < N_REAPPRAISALS:
                # If rate_reap list has not been initialized, question is unanswered 
                remaining[q_label] = True
                break
            
            for i in range(N_REAPPRAISALS):    
                # Go through each reappraisal and check ...
                if q_label not in self.ss['rate_reap'][domain][i].keys():
                    # if the question is not in the dict, then it remains to be asked
                    remaining[q_label] = True
                    break
                elif self.ss['rate_reap'][domain][i][q_label] is None:
                    # if the question is none, then it remains to be asked
                    remaining[q_label] = True
                    break
                else:
                    remaining[q_label] = False
                    
        return remaining
            
    def process_input(self, user_input):
        super().process_input(user_input)
        
    def next_state(self) -> str:
        
        q_completions_cur = self.q_completions(self.ss["cur_domain"])
        remaining_qs_cur = [k for k, v in q_completions_cur.items() if v]
        
        q_completions_other = self.q_completions(self.other_domain)
        remaining_qs_other = [k for k, v in q_completions_other.items() if v]
        
        if not remaining_qs_cur and not remaining_qs_other:
            self.ss["state"] = "end"
            self.ss["show_chat"] = False
            return "end", {}
        elif not remaining_qs_cur:
            self.switch_cur_domain()
            self.ss["state"] = "generate_reaps"
            self.ss["show_chat"] = False
            return "generate_reaps", {}
        elif remaining_qs_cur:
            self.ss["state"] = "rate_reap"
            self.ss["show_chat"] = False
            return "rate_reap", {"next_q_label": remaining_qs_cur[0]}
    
    def slider_callback(self, q_label):
        domain = self.ss["cur_domain"]
        for i in range(N_REAPPRAISALS):
            slider_key = q_label + "_" + domain + "_" + str(i)
            label, _, i = slider_key.split("_")
            i = int(i)
            self.ss["rate_reap"][domain][i][label] = self.st.session_state[slider_key]
    
            db_reap_feature(
                pid=self.ss["pid"],
                issue_type=domain,
                reap_num=i,
                feature_type=label.upper(),
                value=self.st.session_state[slider_key]
            )
            
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
        
        with self.st.form(key="rate_reap"):
            
            # Print the question
            with self.st.chat_message("assistant"):
                self.st.markdown(bot_msgs["rate_reap_stem"], unsafe_allow_html=True)
                # self.st.markdown(bot_msgs["rate_reap"][next_q_label], unsafe_allow_html=True)
            
            q_col, reap_col = self.st.columns([1, 3])
            with q_col:
                self.st.markdown(bot_msgs["rate_reap"][next_q_label], unsafe_allow_html=True)
            with reap_col:
                # Loop through each reappraisal and ask the user to rate it
                for i, reap in enumerate(self.ss["reaps"][self.ss["cur_domain"]]):
                    q_key = next_q_label + "_" + self.ss["cur_domain"] + "_" + str(i)
                    self.st.slider(reap, min_value=0, max_value=100, step=1, key=q_key)
            
            self.st.form_submit_button("Continue", on_click=lambda: self.slider_callback(next_q_label))

        return
    

class BotSummarizeIssue(BotStep):
    
    def __init__(self, st):
        super().__init__(st)
        self.cur_state = "summarize_issue"
        
    def generate_summary(self, issue_msgs: List[dict]):
        prompt = prompts["summarize_issue"]
        msgs = self.ss["issue_msgs"][self.ss["cur_domain"]]
        summary = Chatbot.query_gpt(
            prompt=prompt, 
            messages=msgs,
            pid=self.ss["pid"]  # for logging
            )
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
#         self.cur_state = "survey"
    
#     def process_input(self, user_input):
#         return
        
#     def next_state(self) -> str:
#         return None, None
            
#     def generate_output(self, **kwargs):
#         return 

class BotEnd(BotStep):
    
    def __init__(self, st):
        super().__init__(st)
        self.cur_state = "end"
        
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
