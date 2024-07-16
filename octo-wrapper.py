# This is a wrapper for using octo-models for inference
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from octo.model.octo_model import OctoModel
import requests
import matplotlib.pyplot as plt
import numpy as np

# Global configs
INSTRCUTION = "Enter the task here"
MODEL = "hf://rail-berkeley/octo-small-1.5"

model = OctoModel.load_pretrained(MODEL)