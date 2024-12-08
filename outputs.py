
from typing import List


class Output():
    
    def __init__(self, type):
        
        self.type = type
        
        
class TextOutput(Output):
        
    def __init__(self, text):
        
        super().__init__(type="text")
        self.text = text
        
        
class TextColumnsOutput(Output):
    
    def __init__(self, text: str, col_texts: List[str]):
        
        super().__init__(type="text_columns")
        self.col_texts = col_texts
        self.text = text
        self.num_columns = len(col_texts)
        

class SliderOutput(Output):
    
    def __init__(self, key: str, text: str, start: int, min: int, max: int, step: int=1):
        
        super().__init__(type="slider")
        self.text = text
        self.key = key
        self.start = start
        self.min = min
        self.max = max
        self.step = step


class RadioOutput(Output):
    
    def __init__(self, key: str, text: str, options: List[str], captions=None):
        
        super().__init__(type="radio")
        self.key = key
        self.text = text
        self.options = options
        self.captions = captions


class RankReapsOutput(Output):
    
    def __init__(self, text: str, reaps: List[str]):
        
        super().__init__(type="rank_reaps")
        self.text = text
        self.reaps = reaps


        