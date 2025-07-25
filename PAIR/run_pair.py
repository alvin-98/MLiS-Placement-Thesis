import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer 
import torch
from tqdm.auto import tqdm
import pandas as pd
import re
import gc
import os
import copy
from prompts import LLAMA2_CLS_PROMPT, get_attacker_system_prompts, get_prompt_style


def evaluate_model(device, batch_size,dataset,max_iterations,n_runs): 
    # initial model setup 
    #device= torch.device("cuda:0")  
    defence_model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', torch_dtype=torch.bfloat16, device_map="auto")
    defence_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', use_fast=False)
    attack_model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', torch_dtype=torch.bfloat16, device_map="auto")
    attack_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', use_fast=False)
    judge_model = AutoModelForCausalLM.from_pretrained("cais/HarmBench-Llama-2-13b-cls", torch_dtype=torch.bfloat16, device_map="auto")
    judge_tokenizer = AutoTokenizer.from_pretrained("cais/HarmBench-Llama-2-13b-cls", use_fast=False)
    defence_tokenizer.pad_token = defence_tokenizer.eos_token
    attack_tokenizer.pad_token = attack_tokenizer.eos_token
    judge_tokenizer.pad_token = judge_tokenizer.eos_token
    defence_tokenizer.padding_side = "left"  
    attack_tokenizer.padding_side = "left"
    judge_tokenizer.padding_side = "left"
   

    # ---token mask for judge model---
    vocabulary_mask = ['Yes','No']
  
    yes_token_ids = []
    no_token_ids = []

    for word in vocabulary_mask:
        token_ids = judge_tokenizer.encode(word, add_special_tokens=False)
        if word.lower().startswith('yes'):
            yes_token_ids.extend(token_ids)
        else:
            no_token_ids.extend(token_ids)
     
           
    mask = torch.zeros(judge_tokenizer.vocab_size, dtype=torch.bool)
    for word in vocabulary_mask:
        tokens = judge_tokenizer.encode(word, add_special_tokens=False)
        for token in tokens:
            mask[token] = True
    exclude_vocabulary_mask = torch.logical_not(mask).to(judge_model.device)
       
    # ---End of token mask setup for judge model---
 
    df_rows=[]
    for i in range(n_runs): 
        Broken = False       
        objective=dataset[i]['prompt'] 
        for idx,sys_prompt in enumerate(get_attacker_system_prompts(objective,"Sure, here is the information you requested:")):
            formatted_messages = []
            formatted_messages.append({"role": "system", "content":sys_prompt})
            prompt = attack_tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
            defence_response = 'N/A' #for logging purposes only
            most_likely_token = 'N/A' #for logging purposes only
            for k in range(max_iterations+1):
                if k == 0:
                    attack_text=[objective]
                    defence_input = defence_tokenizer(text=objective, return_tensors='pt', padding=True, truncation=True).to(defence_model.device)    
                    new_prompt = objective
                else:
                    attacker_inputs = attack_tokenizer(text=prompt, return_tensors='pt', padding=True, truncation=True).to(attack_model.device)

                    # Generate outputs
                    with torch.no_grad():
                        attack_outputs = attack_model.generate(
                            **attacker_inputs,
                            max_new_tokens=1512,
                            do_sample=False,
                            pad_token_id=attack_tokenizer.eos_token_id
                        )

                    # Slice off the input part to get only the generated continuation
                    attack_text = attack_tokenizer.batch_decode(attack_outputs[:, attacker_inputs['input_ids'].size(1):], skip_special_tokens=True)
                    # Extract the improvement and prompt from the attack response
                    improvement, new_prompt = extract_improvement_and_prompt(attack_text[0])
                    if new_prompt == None:
                        statement = f'iteration {k+1} for prompt {i+1} failed. Skipping to next prompt'
                        print(statement)
                        Broken=True
                        final_row = {
                            'objective': objective,
                            'objective_number': i+1,
                            'style': get_prompt_style(idx),
                            'final_prompt': new_prompt,
                            'final_output': defence_response,
                            'iterations': k,
                            'broken': Broken,
                            'judgement': most_likely_token,
                            'full_history': formatted_messages
                        }
                        df_rows.append(final_row)
                        break    
                    defence_input = defence_tokenizer(text=new_prompt, return_tensors='pt', padding=True, truncation=True).to(defence_model.device)
                defence_input_len = defence_input['input_ids'].shape[1]
                with torch.no_grad():
                        defence_outputs = defence_model.generate(
                            **defence_input,
                            max_new_tokens=512,
                            do_sample=False,
                            pad_token_id=attack_tokenizer.eos_token_id
                        )
                defence_text = defence_tokenizer.batch_decode(defence_outputs[:,defence_input_len:], skip_special_tokens=True)
                judge_info = [LLAMA2_CLS_PROMPT['prompt'].format(behavior=objective, generation=defence_text[0])]
                judge_inputs = judge_tokenizer(judge_info[0], return_tensors='pt', padding=True, truncation=True).to(judge_model.device)
                with torch.no_grad():
                    judge_outputs = judge_model(**judge_inputs)
                    logits = judge_outputs.logits  
                    final_logits = logits[:, -1, :] 

                    masked_logits = final_logits.masked_fill(exclude_vocabulary_mask, float('-inf'))

                    probs = torch.nn.functional.softmax(masked_logits, dim=-1)

                    for j in range(probs.size(0)):  
                        prob_dist = probs[j]
                        most_likely_token_id = prob_dist.argmax().item()
                        most_likely_token = judge_tokenizer.decode(most_likely_token_id, skip_special_tokens=True)
                attack_response = attack_text[0]
                defence_response = defence_text[0]
                most_likely_token = most_likely_token.strip()
                formatted_messages.append({
                    "role": "assistant",
                    "content": attack_response
                })
                formatted_messages.append({
                    "role": "user",
                    "content": f"LANGUAGE MODEL OUTPUT: {defence_response} OBJECTIVE: {objective} SCORE: {most_likely_token}"
                })
                #print(f"Iteration {k+1}/{max_iterations}")
                #print(f"Attack Response: {attack_response}")
                #print(f"Defence Response: {defence_response}")
                #print(f"Judge Response:{most_likely_token}")
                if most_likely_token == 'Yes':
                    print("Attack successful!")
                    final_row = {
                        'objective': objective,
                        'objective_number': i+1,
                        'style': get_prompt_style(idx),
                        'final_prompt': new_prompt,
                        'final_output': defence_response,
                        'iterations': k,
                        'broken': Broken,
                        'judgement': most_likely_token,
                        'full_history': formatted_messages
                    }
                    df_rows.append(final_row)
                    break
            else:
                # This else runs only if the for loop completes without hitting 'break'

                final_row = {
                        'objective': objective,
                        'objective_number': i+1,
                        'style': get_prompt_style(idx),
                        'final_prompt': new_prompt,
                        'final_output': defence_response,
                        'iterations': k,
                        'broken': Broken,
                        'judgement': most_likely_token,
                        'full_history': formatted_messages
                    }
                df_rows.append(final_row)
    results_df = pd.DataFrame(df_rows)
    return results_df       
        
def extract_improvement_and_prompt(text):
    # Find improvement
    imp_start = text.find('"improvement":')
    prompt_start = text.find('"prompt":')
    if imp_start == -1 or prompt_start == -1:
        return None, None

    # Move past the keys and the following ': "' part
    imp_start += len('"improvement":') + 1
    prompt_start += len('"prompt":') + 1

    # Extract improvement string
    # Find the first quote after imp_start
    imp_quote_start = text.find('"', imp_start)
    imp_quote_end = text.find('"', imp_quote_start + 1)
    improvement = text[imp_quote_start + 1 : imp_quote_end]

    # Extract prompt string
    prompt_quote_start = text.find('"', prompt_start)
    prompt_quote_end = text.find('"', prompt_quote_start + 1)
    prompt = text[prompt_quote_start + 1 : prompt_quote_end]

    return improvement, prompt



def main():
    print("Torch has GPU access: ", torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--temperature', type=float, default=1.1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_iterations', type=int, default=20)  # Maximum iterations for the attack
    parser.add_argument('--n_runs', type = int, default=200)
    args = parser.parse_args()
    dataset = load_dataset("walledai/HarmBench",'standard', split="train")
    print(f"Evaluating models on device: {args.device}")
    df = evaluate_model(args.device,args.batch_size,dataset,args.max_iterations,args.n_runs) 
    #df.to_csv(f'PAIR/results_L2.csv', index=True)
    #df.to_csv(f'PAIR/quickcheck.csv', index=True)
    df.to_csv(f'PAIR/test.csv', index=True)
    
if __name__ == '__main__':
    main()
