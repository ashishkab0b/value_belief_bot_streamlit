
class DevelopmentConfig:
    
    def __init__(self):
        
        self.llm_model = "gpt-4o"
        self.temperature = 1.0
        self.dynamo_tbl_name = "value_belief_chatbot_v1"
        self.n_reappraisals = 5
        self.debug = True
        self.domains = ["career", "relationship"]
        self.error_message = "I'm sorry, there has been an error. Please contact the researcher on Prolific."
        
        
class Config(DevelopmentConfig):
    
    def __init__(self):
        
        super().__init__()