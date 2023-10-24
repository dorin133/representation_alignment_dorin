from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np

from repe import repe_pipeline_registry
repe_pipeline_registry()


model_name_or_path = "/cs/labs/shashua/noamw02/llama_weights_hf/llama-2-7b-chat/"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", token=True).eval()
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True)
tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
tokenizer.bos_token_id = 1




######Inject h(word1)-h(word2) and let it propagate to final layer
from repe.rep_control_reading_vec import WrappedReadingVecModel

def word_rep(word):
    tokens = torch.unsqueeze(torch.tensor(tokenizer(word)['input_ids']), dim=0)
    with torch.no_grad():
        output = model(tokens.to(model.device), output_hidden_states=True)
    hidden_states = output.hidden_states  # first index is the layer number, second index is trivial, third is the token index
    return(hidden_states)


word_good = "good"#"happy"
word_bad = "bad"#"sad"
hidden_good = word_rep(word_good)
hidden_bad = word_rep(word_bad)

delta_h = []
for i in range(-1,-20,-1): #layers
    print(i)
    layer_id = [i]
    block_name="decoder_block"

    wrapped_model = WrappedReadingVecModel(model, tokenizer)
    wrapped_model.unwrap()
    wrapped_model.wrap_block(layer_id, block_name=block_name)

    #get normalized direction
    v = hidden_good[i][0][-1] - hidden_bad[i][0][-1]
    v = v/torch.norm(v)
    v.cpu()

    coeff=2.0

    activations = {}
    for layer in layer_id:
        activations[layer] = torch.tensor(coeff * v).to(model.device).half()
        print("injected vector norm = "+str(torch.norm(activations[layer])))

    prompt = "[INST]Hello, how are you?[\INST]"
    prompt_tokens = torch.unsqueeze(torch.tensor(tokenizer(prompt)['input_ids']),dim=0)

    wrapped_model.reset()
    wrapped_model.set_controller(layer_id, activations, block_name)
    with torch.no_grad():
        output = wrapped_model(prompt_tokens.to(model.device),output_hidden_states=True)
    hidden_state_controlled = output.hidden_states[-1][0][-1].cpu() #first index is the layer number, second index is trivial, third is the token index

    wrapped_model.reset()
    with torch.no_grad():
        output = wrapped_model(prompt_tokens.to(model.device),output_hidden_states=True)
    hidden_state = output.hidden_states[-1][0][-1].cpu() #first index is the layer number, second index is trivial, third is the token index
    delta_h.append(torch.norm(hidden_state_controlled - hidden_state).item())

print(delta_h)
