# import torch
import tokenize_script

device = 'cpu'

text = tokenize_script.tokenize(["a new", "a old", "a cat"]).to(device)

print("text tokens:", text)  # prints: [[0.9927937  0.00421068 0.00299572]]