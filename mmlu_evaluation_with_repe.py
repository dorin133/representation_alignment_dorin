import csv
import tqdm
import torch
import pandas as pd
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from repe.rep_control_reading_vec import WrappedReadingVecModel


def load_mmlu_dataset(mmlu_dataset_name):
    return pd.read_csv(f'/cs/usr/noamw02/labs/mmlu/data/test/{mmlu_dataset_name}_test.csv')



def test_model(model, tokenizer, mmlu_dataset_name):
    score = 0
    letter_to_number = {'A':0, 'B':1, 'C':2, 'D':3}
    logit_sum = 0
    for i in tqdm.trange(len(dataset)):
        example = dataset.iloc[i]

        instruction = f'{example[0]}\nA. {example[1]}\nB. {example[2]}\nC. {example[3]}\nD. {example[4]}\n The letter of the correct answer is '
        prompt = torch.unsqueeze(torch.tensor(tokenizer.encode(tokenizer.encode(instruction))),dim=0)
        label = example[-1]
        label_number = letter_to_number[label]
        with torch.no_grad():
            logits = model(input_ids=prompt.cuda()).logits[0, -1]
            res_answer = logits[[319, 350, 315, 360]]
            prediction = torch.argmax(res_answer).cpu().item()
        if prediction==label_number:
            score += 1
        logit_sum += res_answer[label_number]
    accuracy = score/len(dataset)
    logit_mean = logit_sum/len(dataset)
    print(mmlu_dataset_name+': '+str(accuracy))
    print(mmlu_dataset_name+' logit mean: '+str(logit_mean))

    return(accuracy,logit_mean)


def word_rep(word):
    tokens = torch.unsqueeze(torch.tensor(tokenizer(word)['input_ids']), dim=0)
    with torch.no_grad():
        output = model(tokens.to(model.device), output_hidden_states=True)
    hidden_states = output.hidden_states  # first index is the layer number, second index is trivial, third is the token index
    return (hidden_states)

if __name__ == '__main__':

    #load dataset
    mmlu_dataset_name = 'clinical_knowledge'#other options: 'clinical_knowledge'#'college_chemistry','international_law','global_facts','philosophy','high_school_computer_science','clinical_knowledge'
    dataset = load_mmlu_dataset(mmlu_dataset_name)

    #load model
    model_name_or_path = "/cs/labs/shashua/noamw02/llama_weights_hf/llama-2-7b-chat/"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto",
                                                 token=True).eval()
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left",
                                              legacy=False, token=True)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1

    #calculate diff in word representations
    word_good = "good"  #other options: "happy"
    word_bad = "bad"  #other options: "sad"
    hidden_good = word_rep(word_good)
    hidden_bad = word_rep(word_bad)


    #prepare RepE model
    block_name = "decoder_block"
    layer_id = list(range(8, 32, 3))
    wrapped_model = WrappedReadingVecModel(model, tokenizer)
    wrapped_model.unwrap()
    wrapped_model.wrap_block(layer_id, block_name=block_name)

    #test model on dataset for various norms of injected vectors
    scores = np.zeros(10) # % of correct answers
    logits = np.zeros(10) # log probability of the correct answer
    for j, coeff in enumerate(range(0, 10, 1)):
        activations = {}
        for layer in layer_id:
            v = hidden_good[layer][0][-1] - hidden_bad[layer][0][-1]
            v = v / torch.norm(v)
            v.cpu()
            activations[layer] = torch.tensor(coeff * v).to(model.device).half()

        wrapped_model.reset()
        wrapped_model.set_controller(layer_id, activations, block_name)
        scores[j], logits[j] = test_model(wrapped_model, tokenizer, mmlu_dataset_name)
