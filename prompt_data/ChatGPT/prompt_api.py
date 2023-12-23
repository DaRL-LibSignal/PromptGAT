import openai
import numpy as np
import random
import re
from functools import lru_cache

# openai.api_key = my_key

class Conversation:
    def __init__(self, api_key):
        openai.api_key = api_key
      
    def set_values(self, veichle_num, capacity):
        self.veichle_num = veichle_num
        self.capacity = capacity
      
    
    def get_weather(self):
        choice = ['heavy snowy', 'middle snowy', 'snowy with rain', 'sunny'] 
        value = [0.25, 0.5, 0.75, 0.99]
        index = random.randint(0, len(choice)-1)
        weather_type = choice[index]
        weather_value = value[index]

        return weather_type, weather_value

    def get_response(self, prompt_info, max_token, model):
        knowledge_respon = openai.Completion.create(
          model = model,
          max_tokens = max_token,
          prompt = prompt_info
        )
        # print(knowledge_respon)
        return knowledge_respon

    def get_response_text(self, origin_response):
        return origin_response['choices'][0]['text']

def check_string(string):
    pattern = r'^\d.*\..*\..*$'
    if re.match(pattern, string):
        return True
    else:
        return False

def split_data(raw):
    line_array = raw.replace("[", "").replace("]", "").split(",")
    record_value = []
    for i in range(len(line_array)):
        # print(line_array[i])
        result = line_array[i].split(':')[1].strip()

        if check_string(result):
            result = result[:-1]
        if result.startswith("{") and result.endswith("}"):
            result = result.replace("{", "").replace("}", "")

        result = float(result)
        record_value.append(result)
    
    return record_value

# @lru_cache()
def save_prompt_data(data):
    path = "./prompt_data/ChatGPT/collect_data/experience.txt"
    with open(path, mode='a+') as save_e:
        save_e.write(data)
        save_e.write("\n")

    