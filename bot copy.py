from logger_setup import setup_logger
import yaml
from langchain.schema import HumanMessage, AIMessage
from typing import Literal, Dict, Any, List, Tuple
from langchain_core.runnables import chain
import json
import asyncio
import random
from decimal import Decimal
from datetime import datetime
from itertools import product
from crud import (
    db_add_value, 
    db_add_reappraisal,
    db_set_state, 
    db_set_emotions, 
    db_get_state, 
    db_get_vals, 
    db_get_messages, 
    db_get_issue_messages, 
    db_get_reappraisals, 
    db_get_emotions,
    db_add_neg_rating,
    db_add_pos_rating,
    db_update_convo_completion,
    db_get_pid
)
from runnables import (
    explain_emotions, 
    generate_value_reap, 
    generate_general_reap, 
    generate_n_reaps,
    identify_values,
    identify_primals,
    identify_value,
    identify_primal
    )



logger = setup_logger()


with open("bot.yml", "r") as ymlfile:
    bot_data = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
with open("messages.yml", "r") as ymlfile:
    msgs = yaml.load(ymlfile, Loader=yaml.FullLoader)
   


'''
Each stage of the bot needs 
- a function to generate the bot message
- a function to determine the next state

there's different kinds of triggers for the next state

Depending on state during last message, 
get a particular bot class. That bot will make a decision for what to call 
next. it will call the next one and generate messages.


I will have next_bot return something
that will get passed to the message generation class
often it won't matter, but sometimes it can
'''

N_REAPPRAISALS = 5

class Bot:

    def __init__(self, chat_id, prev_state):
        self.chat_id = chat_id
        self.prev_state = prev_state
        if self.prev_state != self.cur_state:
            db_set_state(self.chat_id, self.cur_state)
    
    
    def next_bot(self):
        '''
        A function that returns the next bot class to instantiate for next action'''
        raise NotImplementedError
    
    
    def generate_messages(self, request_data, **kwargs) -> List[Dict[str, Any]]:
        '''A function that generates the bot messages'''
        raise NotImplementedError
        

class BotBegin(Bot):
    
    def __init__(self, chat_id, prev_state):
        self.cur_state = "begin"
        super().__init__(chat_id=chat_id, prev_state=prev_state)
    
    def next_bot(self, request_data) -> Tuple[Bot, Dict[str, Any]]:
        '''return the class of the next bot to generate messages'''
        return BotHello(self.chat_id, self.cur_state), {}
    
    def generate_messages(self, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError


class BotHello(Bot):
    
    def __init__(self, chat_id, prev_state):
        self.cur_state = "hello"
        super().__init__(chat_id=chat_id, prev_state=prev_state)
        
    def next_bot(self, request_data) -> Tuple[Bot, Dict[str, Any]]:
        return BotSolicitIssue(self.chat_id, self.cur_state), {}
    
    def generate_messages(self, **kwargs) -> List[Dict[str, Any]]:
        resp = {
            "sender": "bot",
            "response": msgs["introduction"],
            "widget_type": "text",
            "widget_config": {}
        }
        return [resp]
        

class BotSolicitIssue(Bot):
    
    def __init__(self, chat_id, prev_state):
        self.cur_state = "solicit_issue"
        super().__init__(chat_id=chat_id, prev_state=prev_state)
        
    def next_bot(self, request_data) -> Tuple[Bot, Dict[str, Any]]:
        return BotSolicitEmotions(self.chat_id, self.cur_state), {}
    
    def generate_messages(self, **kwargs) -> List[Dict[str, Any]]:
        resp = {
            "sender": "bot",
            "response": msgs["solicit_issue"],
            "widget_type": "text",
            "widget_config": {}
        }
        return [resp]
    
class BotSolicitEmotions(Bot):
    
    def __init__(self, chat_id, prev_state):
        self.cur_state = "solicit_emotions"
        super().__init__(chat_id=chat_id, prev_state=prev_state)
        
        
    def next_bot(self, request_data) -> Tuple[Bot, Dict[str, Any]]:
        # Parse and store selected emotions
        emotions = request_data.get("response", [])
        # emotions = [{"emotion": emo} for emo in emotions]
        db_set_emotions(self.chat_id, emotions)
        return BotExplainEmotions(self.chat_id, self.cur_state), {}
    
    
    def generate_messages(self, **kwargs) -> List[Dict[str, Any]]:
        resp = {
        "sender": "bot",
        "response": msgs["solicit_emotions"],
        "widget_type": "multiselecttext",
        "widget_config": {
            "options": bot_data["emotions"]
            }
        }
        return [resp]
    

class BotExplainEmotions(Bot):
    
    def __init__(self, chat_id, prev_state):
        self.cur_state = "explain_emotions"
        super().__init__(chat_id=chat_id, prev_state=prev_state)
        
        
    def next_bot(self, request_data) -> Tuple[Bot, Dict[str, Any]]:
        '''if llm says finished, return next bot
        otherwise return the message that the llm provided 
        which will serve as the next message'''
        llm_msg = self.get_llm_response()
        if "::finished::" in llm_msg:
            db_set_state(self.chat_id, "rate_neg")
            return BotRateNeg(self.chat_id, self.cur_state), {}
        return BotExplainEmotions(self.chat_id, self.cur_state), {"msg": llm_msg}
    
    def get_llm_response(self):
        '''Gets the next message from the llm (explain emotions) or the stop signal'''
        messages = db_get_messages(self.chat_id)
        lc_history = convert_to_lc_history(messages)
        emotions = db_get_emotions(self.chat_id)
        # selected_emotions = [emo.get("emotion") for emo in emotions if emo.get("emotion")]
        msg = explain_emotions.invoke({"messages": lc_history, "emotions": emotions})
        return msg
    
    def generate_messages(self, **kwargs) -> List[Dict[str, Any]]:
        msg = kwargs.get('msg', self.get_llm_response())
        resp = {
            "sender": "bot",
            "response": msg,
            "widget_type": "text",
            "widget_config": {
                "metadata": {
                    "msg_type": "explain_emotions"
                }
            }
        }
        return [resp]
    
class BotRateNeg(Bot):
        
        def __init__(self, chat_id, prev_state):
            self.cur_state = "rate_neg"
            super().__init__(chat_id=chat_id, prev_state=prev_state)
            
        def next_bot(self, request_data) -> Tuple[Bot, Dict[str, Any]]:
            db_add_neg_rating(self.chat_id, request_data.get("response"))
            return BotRatePos(self.chat_id, self.cur_state), {}
        
        def generate_messages(self, **kwargs) -> List[Dict[str, Any]]:
            resp = {
                "sender": "bot",
                "response": msgs["rate_neg"],
                "widget_type": "slider",
                "widget_config": {
                    "metadata": {
                        "msg_type": "rate_neg"
                    },
                    "min": 0,
                    "max": 100,
                    "start": 50,
                    "step": 1
                }
            }
            return [resp]

class BotRatePos(Bot):
    
    def __init__(self, chat_id, prev_state):
        self.cur_state = "rate_pos"
        super().__init__(chat_id=chat_id, prev_state=prev_state)
        
    def next_bot(self, request_data) -> Tuple[Bot, Dict[str, Any]]:
        db_add_pos_rating(self.chat_id, request_data.get("response"))
        return BotSolicitValues(self.chat_id, self.cur_state), {}
    
    def generate_messages(self, **kwargs) -> List[Dict[str, Any]]:
        resp = {
            "sender": "bot",
            "response": msgs["rate_pos"],
            "widget_type": "slider",
            "widget_config": {
                "metadata": {
                    "msg_type": "rate_pos"
                },
                "min": 0,
                "max": 100,
                "start": 50,
                "step": 1
            }
        }
        return [resp]
        
    
class BotSolicitValues(Bot):
    
    def __init__(self, chat_id, prev_state):
        self.cur_state = "solicit_values"
        super().__init__(chat_id=chat_id, prev_state=prev_state)
        
        
    def next_bot(self, request_data) -> Tuple[Bot, Dict[str, Any]]:
        # Save values to database
        value_num = int(request_data['widget_config']['metadata'].get("val_num"))
        value_rating = int(request_data.get("response"))
        value_text = bot_data["vals"][value_num]
        db_add_value(self.chat_id, value_text, value_num, value_rating)
        
        user_vals = db_get_vals(self.chat_id)
        
        if len(user_vals) < len(bot_data["vals"]):
            return BotSolicitValues(self.chat_id, self.cur_state), {"user_vals": user_vals}
        else:
            return BotReappraisal(self.chat_id, self.cur_state), {}
    
    
    def generate_messages(self, **kwargs) -> List[Dict[str, Any]]:
        vals = kwargs.get("user_vals", db_get_vals(self.chat_id))
        done_val_nums = [int(val.get("value_num")) for val in vals if val.get("value_num") is not None]  # Get value numbers already collected
        remaining = [int(i) for i in range(len(bot_data['vals'])) if int(i) not in done_val_nums]  # Get value numbers not collected
        if remaining:
            val_num = random.choice(remaining)
            resp = {
                "sender": "bot",
                "response": msgs["solicit_values"].format(value=bot_data["vals"][val_num]),
                "widget_type": "slider",
                "widget_config": {
                    "min": 0,
                    "max": 100,
                    "start": 50,
                    "step": 1,
                    "metadata": {
                        "val_num": val_num
                    }
                }
            }
            return [resp]
        else:
            return {"error": "All values have been collected"}
    
class BotGenerateJudgeReaps(Bot):
    
    def __init__(self, chat_id, prev_state):
        self.cur_state = "generate_judge_reaps"
        super().__init__(chat_id=chat_id, prev_state=prev_state)
        self.reaps = None
        with open("primals.json", "r") as jsonfile:
            self.primals = json.load(jsonfile)
        with open("bot.yml", "r") as ymlfile:
            self.bot_data = yaml.load(ymlfile, Loader=yaml.FullLoader)
            self.vals = bot_data["vals"]
        
    def next_bot(self, request_data) -> Tuple[Bot, Dict[str, Any]]:
        pass
    
    def generate_reaps(self):
        # Get issue messages
        issue_messages = db_get_issue_messages(self.chat_id)
        
        # Convert chat history to langchain format
        lc_history = convert_to_lc_history(issue_messages)

        # Generate reappraisals
        reaps = generate_n_reaps.invoke({"messages": lc_history, "n": N_REAPPRAISALS})
        self.reaps = reaps
        return reaps
    
    async def judge_reap_val(self, reap, val):
        val_present = identify_value.invoke({"reappraisal": reap, "val": val})
        return val_present
    
    async def judge_reap_belief(self, reap, belief_name, belief_description):
        belief_present = identify_primal.invoke({"reappraisal": reap, "belief_name": belief_name, "belief_description": belief_description})
        return belief_present
    

    async def judge_reaps(self):
        assert self.reaps is not None, "Reappraisals have not been generated"
        
        primals = self.primals
        vals = self.vals
        reaps = {reap_num: {"reap": reap} for reap_num, reap in enumerate(self.reaps)}
        
        for reap in reaps.values():
            reap_text = reap["reap"]
            # Create both val_tasks and belief_tasks as a combined list
            tasks = [
                self.judge_reap_val(reap_text, val) for val in vals
            ] + [
                self.judge_reap_belief(reap_text, primal["belief"], primal["description"]) for primal in primals
            ]
            # Run both sets of tasks concurrently and store the results
            results = await asyncio.gather(*tasks)
            # Separate results for `val_results` and `belief_results`
            reap['val_results'] = results[:len(vals)]
            reap['belief_results'] = results[len(vals):]
        
        return reaps

    def run_judge_reaps(self):
        results = asyncio.run(self.judge_reaps())
        return results
    
    def parse_results(self, results):
        # make a dict for each reap with the reap text, vals present in plain text, beliefs present in plain text, vals absent, beliefs absent
        parsed_results = []
        for reap_num, reap_results in results.items():
            reap_text = reap_results['reap']
            val_results = reap_results['val_results']
            belief_results = reap_results['belief_results']
            val_present = [val for val, present in zip(self.vals, val_results) if present]
            belief_present = [primal['belief'] for primal, present in zip(self.primals, belief_results) if present]
            val_absent = [val for val, present in zip(self.vals, val_results) if not present]
            belief_absent = [primal['belief'] for primal, present in zip(self.primals, belief_results) if not present]
            parsed_results.append({
                "reap_text": reap_text,
                "val_present": val_present,
                "belief_present": belief_present,
                "val_absent": val_absent,
                "belief_absent": belief_absent
            })
        return parsed_results
            


    
    # async def judge_reap_belief(self, reap, belief_name, belief_description):
    #     belief_present = identify_primal.invoke({"reappraisal": reap, "belief_name": belief_name, "belief_description": belief_description})
    #     return belief_present
    
    # async def judge_reaps(self):
    #     assert self.reaps is not None, "Reappraisals have not been generated"
    #     with open("bot.yml", "r") as ymlfile:
    #         bot_data = yaml.load(ymlfile, Loader=yaml.FullLoader)
    #         vals = bot_data["vals"]
    #     with open("primals.json", "r") as jsonfile:
    #         primals = json.load(jsonfile)
    #     reaps = {reap_num: {"reap": reap} for reap_num, reap in enumerate(self.reaps)}
    #     for reap in reaps.values():
    #         reap_text = reap["reap"]
    #         reap['val_tasks'] = [self.judge_reap_val(reap_text, val) for val in vals]
    #         reap['belief_tasks'] = [self.judge_reap_belief(reap_text, primal["belief"], primal["description"]) for primal in primals]
    #     await asyncio.gather(*[task for reap in reaps.values() for task in reap['val_tasks'] + reap['belief_tasks']])
    
    # def run_judge_reaps(self):
    #     results = asyncio.run(self.judge_reaps())
        
    #     return results
    
        
    
    
    
class BotReappraisal(Bot):
        
        def __init__(self, chat_id, prev_state):
            self.cur_state = "reappraisal"
            super().__init__(chat_id=chat_id, prev_state=prev_state)
            
        def next_bot(self, request_data) -> Tuple[Bot, Dict[str, Any]]:
            finished_reaps = db_get_reappraisals(self.chat_id)
            if len(finished_reaps) < 3:
                return BotReappraisal(self.chat_id, self.cur_state), {"finished_reaps": finished_reaps}
            else:
                return BotRankReappraisals(self.chat_id, self.cur_state), {}
        
        def generate_messages(self, **kwargs) -> List[Dict[str, Any]]:
            # new version: generate n reappraisals, then score them for values and beliefs
            # introduce reappraisal if switching states
            if self.prev_state == "solicit_values":
                resp = {
                    "sender": "bot",
                    "response": msgs['intro_reappraisal'],
                    "widget_type": "text",
                    "widget_config": {}
                }
            
                return [resp]
            
            # max_val_dict, min_val_dict = get_max_min_person_value(self.chat_id)
            # finished_reaps = kwargs.get("finished_reaps")
            # finished_conditions = [reap.get("value_rank") for reap in finished_reaps]  # max, min, general
            # unfinished_conditions = [cond for cond in ["max", "min", "general"] if cond not in finished_conditions]
            # next_condition = random.choice(unfinished_conditions)
            # next_reap_num = len(finished_reaps) + 1
            # # Get issue messages
            # issue_messages = db_get_issue_messages(self.chat_id)
            # lc_history = convert_to_lc_history(issue_messages)
            # # Generate a reappraisal message
            # if next_condition == "max":
            #     reap = generate_value_reap.invoke({"messages": lc_history, "value": max_val_dict})
            # elif next_condition == "min":
            #     reap = generate_value_reap.invoke({"messages": lc_history, "value": min_val_dict})
            # elif next_condition == "general":
            #     reap = generate_general_reap.invoke({"messages": lc_history})
            # reap = reap.replace("\n", "<br>")
            
            # # Add the reappraisal to the database
            # if next_condition == "max":
            #     value_dict = max_val_dict
            # elif next_condition == "min":
            #     value_dict = min_val_dict
            # else:
            #     value_dict = None
            # reap_dict = {
            #     "reap_text": reap,
            #     "reap_num": next_reap_num,
            #     "value_text": value_dict.get("value_text") if value_dict else "",
            #     "value_rank": next_condition,
            #     "value_rating": value_dict.get("value_rating") if value_dict else ""
            # }
            # db_add_reappraisal(self.chat_id, **reap_dict)
            
            # # Compose the bot response
            # resp = [{
            #     "sender": "bot",
            #     "response": f"Perspective {next_reap_num} of 3:",
            #     "widget_type": "text",
            #     "widget_config": {}
            # }, 
            # {
            #     "sender": "bot",
            #     "response": reap,
            #     "widget_type": "text",
            #     "widget_config": {
            #         "metadata": {
            #             "msg_type": "reappraisal",
            #             "condition": next_condition}
            #     }
            # }, 
            # {
            #     "sender": "bot",
            #     "response": f"Respond \"ok\" to continue.",
            #     "widget_type": "text",
            #     "widget_config": {}
            # }, 
            # ]
            # return resp


class BotRankReappraisals(Bot):
    
    def __init__(self, chat_id, prev_state):
        self.cur_state = "rank_reappraisals"
        super().__init__(chat_id=chat_id, prev_state=prev_state)
    
    def next_bot(self, request_data) -> Tuple[Bot, Dict[str, Any]]:
        # Retrieve reappraisals from the database
        reaps = db_get_reappraisals(self.chat_id)
        
        # Save the user's judgment to the database
        if self.prev_state == "rank_reappraisals":
            logger.debug('saving reappraisal ranks')
            for reap_num, reap_rank in request_data.get("response").items():
                reap_num = int(reap_num)
                reap_entry = [reap for reap in reaps if reap.get("reap_num") == reap_num][0]
                reap_entry['reap_rank'] = int(reap_rank)
                logger.debug(f'reap_entry: {reap_entry}')
                db_add_reappraisal(**reap_entry)
                
        return BotJudgeReappraisals(self.chat_id, self.cur_state), {}
    
    def generate_messages(self, **kwargs) -> List[Dict[str, Any]]:
        reaps = db_get_reappraisals(self.chat_id)
        reap_dict = {str(reap.get("reap_num")): str(reap.get("reap_text")) for reap in reaps}
        # sorted_reap_list = sorted(reaps, key=lambda x: int(x['reap_num']))
        # reap_texts = [item['reap_text'] for item in sorted_reap_list]

        
        resp = {
            "sender": "bot",
            "response": msgs["rank_reappraisals"],
            "widget_type": "ranking",
            "widget_config": {
                "metadata": {
                    "msg_type": "rank_reappraisals",
                },
                "item_dict": reap_dict
                # "itemsList": reap_texts
            }
        }
        return [resp]


class BotJudgeReappraisals(Bot):
    
    def __init__(self, chat_id, prev_state):
        self.cur_state = "judge_reappraisals"
        super().__init__(chat_id=chat_id, prev_state=prev_state)
        
    def next_bot(self, request_data) -> Tuple[Bot, Dict[str, Any]]:
        
        # Retrieve reappraisals from the database
        reaps = db_get_reappraisals(self.chat_id)
        
        # Save the user's judgment to the database
        if request_data['widget_config']['metadata']['msg_type'] == "reappraisal_success":
            prev_reap_num = int(request_data['widget_config']['metadata'].get("reap_num"))
            prev_reap_efficacy = int(request_data.get("response"))
            prev_reap_entry = [reap for reap in reaps if reap.get("reap_num") == prev_reap_num][0]
            assert "reap_efficacy" not in prev_reap_entry.keys(), "Reappraisal efficacy already judged"
            db_add_reappraisal(reap_efficacy=prev_reap_efficacy, **prev_reap_entry)
        
        # Retrieve reappraisals from the database ( could eliminate this extra call TODO)
        reaps = db_get_reappraisals(self.chat_id)
        # Identify which reappraisals have been judged
        unjudged_reaps = [reap for reap in reaps if reap.get("reap_efficacy") is None]
        
        # Return next bot
        if len(unjudged_reaps) == 0:
            return BotGoodbye(self.chat_id, self.cur_state), {}
        return BotJudgeReappraisals(self.chat_id, self.cur_state), {"unjudged_reaps": unjudged_reaps}
    
    def generate_messages(self, **kwargs) -> List[Dict[str, Any]]:
        # if switching states, introduce next phase
        if self.prev_state != "judge_reappraisals":
            resp = {
                "sender": "bot",
                "response": msgs["intro_judge_reappraisals"],
                "widget_type": "text",
                "widget_config": {
                    "metadata": {
                        "msg_type": "intro_judge_reappraisals"
                    }
                }
            }
            return [resp]
        
        unjudged_reaps = kwargs.get("unjudged_reaps")
        cur_reap = unjudged_reaps[0]
        reap_text = cur_reap.get("reap_text")
        reap_num = cur_reap.get("reap_num")
        condition = cur_reap.get("value_rank")
        resp = [
            {
                "sender": "bot",
                "response": f"Perspective {reap_num} of 3:",
                "widget_type": "text",
                "widget_config": {}
            },
            {
                "sender": "bot",
                "response": reap_text,
                "widget_type": "text",
                "widget_config": {}
            },
            {
                "sender": "bot",
                "response": msgs["reappraisal_success"],
                "widget_type": "slider",
                "widget_config": {
                    "metadata": {
                        "msg_type": "reappraisal_success",
                        "condition": condition,
                        "reap_num": reap_num},
                    "min": 0,
                    "max": 100,
                    "start": 50,
                    "step": 1
                }
        }]
        return resp
        

class BotGoodbye(Bot):
    
    def __init__(self, chat_id, prev_state):
        self.cur_state = "finished"
        super().__init__(chat_id=chat_id, prev_state=prev_state)

    def next_bot(self, request_data: dict) -> Tuple[Bot, Dict[str, Any]]:
        return None, {}
    
    def generate_messages(self, **kwargs) -> List[Dict[str, Any]]:
        pid = db_get_pid(self.chat_id)
        resp = {
            "sender": "bot",
            "response": msgs["finished"].format(pid=pid),
            "widget_type": "text",
            "widget_config": {}
        }
        db_update_convo_completion(self.chat_id, 1)
        return [resp]


def get_max_min_person_value(chat_id):
    vals = db_get_vals(chat_id)
    assert len(vals) == len(bot_data['vals']), "Not all values have been collected"
    val_ratings = [val.get("value_rating") for val in vals if val.get("value_rating") is not None]
    max_val = max(val_ratings)
    min_val = min(val_ratings)
    val_rating_max_indices = [i for i, rating in enumerate(val_ratings) if rating == max_val]
    val_rating_min_indices = [i for i, rating in enumerate(val_ratings) if rating == min_val]
    val_rating_max_idx = random.choice(val_rating_max_indices)
    val_rating_min_idx = random.choice(val_rating_min_indices)
    max_val_dict = vals[val_rating_max_idx]
    min_val_dict = vals[val_rating_min_idx]
    return max_val_dict, min_val_dict


def convert_to_lc_history(chat_history):
    '''
    Convert chat history to langchain format
    '''
    lc_history = []
    for msg in chat_history:
        if msg["sender"] == "bot":
            lc_history.append(AIMessage(str(msg["response"])))
        else:
            lc_history.append(HumanMessage(str(msg["response"])))
    return lc_history


    
def process_next_step(chat_id, request_data):
    logger.debug(f'Processing next step for chat_id: {chat_id}')
    prev_state = db_get_state(chat_id)
    # logger.debug(f'chat_id: {chat_id}')
    logger.debug(f'prev_state: {prev_state}')
    logger.debug(f'user message: {request_data["response"]}')
    bots = {
        "begin": BotBegin,
        "hello": BotHello,
        "solicit_issue": BotSolicitIssue,
        "solicit_emotions": BotSolicitEmotions,
        "explain_emotions": BotExplainEmotions,
        "rate_neg": BotRateNeg,
        "rate_pos": BotRatePos,
        "generate_judge_reaps": BotGenerateJudgeReaps,
        "solicit_values": BotSolicitValues,
        "solicit_beliefs": BotSolicitBeliefs,
        "reappraisal": BotReappraisal,
        "rank_reappraisals": BotRankReappraisals,
        "judge_reappraisals": BotJudgeReappraisals
    }
    # We call the next_bot method of the bot class 
    # to for the message the user just responded to
    prev_bot = bots[prev_state](chat_id, prev_state)
    next_bot, kw = prev_bot.next_bot(request_data)
    msgs = next_bot.generate_messages(**kw)
    response = {"messages": msgs}
    logger.debug(f'bot response: {response}')
    return response
