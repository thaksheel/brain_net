import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from typing import Literal, List, Optional, Dict 

from . import * 


class Runner: 
    def __init__(self, params: Params):
        self.params = params 

    def run_adhd(
        self,
        methods: List[str],
        altas: List[str],
        grid_search: bool = True, 
        exports: bool = True,
    ):
        pass 

    def run_hcp(self):
        pass 
