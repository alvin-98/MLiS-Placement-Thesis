import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm
import pandas as pd

def top_k(model,input_ids, attention_mask, args):
    """
    Top-k sampling decoding: sample from the top-k most probable tokens.
    """
    eos_id = model.config.eos_token_id
    device = input_ids.device
    generated = input_ids
    mask = attention_mask
    past = None
    for _ in range(args.max_new_tokens):
        if past is None:
            out = model(
                input_ids=generated,
                attention_mask=mask,
                use_cache=True
            )
        else:
            out = model(
                input_ids=generated[:, -1:],
                attention_mask=mask,
                past_key_values=past,
                use_cache=True
            )
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        # apply temperature
        logits = logits / args.temperature
        # top-k sampling
        topk_vals, topk_idx = torch.topk(logits, args.k, dim=-1)
        probs = torch.zeros_like(logits).scatter_(1, topk_idx, torch.softmax(topk_vals, dim=-1))
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)
        mask = torch.cat([
            mask,
            torch.ones((generated.size(0), 1), dtype=mask.dtype, device=device)
        ], dim=-1)
        if next_token.item() == eos_id:
            break
    return generated[:, input_ids.size(1):]

def top_p_sorter(logits,p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    cutoff = (cumulative_probs > p).nonzero(as_tuple=True)[1][0] + 1

    top_p_probs = sorted_probs[:, :cutoff]
    top_p_indices = sorted_indices[:, :cutoff]

    return top_p_probs, top_p_indices

def top_p(model,input_ids, attention_mask, args):
    """
    Top-p sampling decoding: sample from the top-p most probable tokens.
    """
    eos_id = model.config.eos_token_id
    device = input_ids.device
    generated = input_ids
    mask = attention_mask
    past = None
    for _ in range(args.max_new_tokens):
        if past is None:
            out = model(
                input_ids=generated,
                attention_mask=mask,
                use_cache=True
            )
        else:
            out = model(
                input_ids=generated[:, -1:],
                attention_mask=mask,
                past_key_values=past,
                use_cache=True
            )
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        # apply temperature
        logits = logits / args.temperature
        # top-k sampling
        topp_vals, topp_idx = top_p_sorter(logits, args.p)
        probs = torch.zeros_like(logits).scatter_(1, topp_idx, torch.softmax(topp_vals, dim=-1))
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)
        mask = torch.cat([
            mask,
            torch.ones((generated.size(0), 1), dtype=mask.dtype, device=device)
        ], dim=-1)
        if next_token.item() == eos_id:
            break
    return generated[:, input_ids.size(1):]


    

def greedy_search(model, input_ids, attention_mask, args):
    """
    Greedy decoding: pick the highest-probability token at each step.
    """
    eos_id = model.config.eos_token_id
    device = input_ids.device
    generated = input_ids
    mask = attention_mask
    past = None
    for _ in range(args.max_new_tokens):
        if past is None:
            out = model(
                input_ids=generated,
                attention_mask=mask,
                use_cache=True
            )
        else:
            out = model(
                input_ids=generated[:, -1:],
                attention_mask=mask,
                past_key_values=past,
                use_cache=True
            )
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        mask = torch.cat([
            mask,
            torch.ones((generated.size(0), 1), dtype=mask.dtype, device=device)
        ], dim=-1)
        if next_token.item() == eos_id:
            break
    return generated[:, input_ids.size(1):]

def beam_search(model, input_ids, attention_mask, args):
    """
    Beam search decoding with num_beams.
    """
    eos_id = model.config.eos_token_id
    device = input_ids.device
    beam_size = args.num_beams
    seq_len = input_ids.size(1)
    # initialize beam candidates
    beams = [{
        'ids': input_ids,
        'mask': attention_mask,
        'past': None,
        'score': 0.0,
        'done': False
    }]
    for _ in range(args.max_new_tokens):
        all_candidates = []
        for beam in beams:
            if beam['done']:
                all_candidates.append(beam)
                continue
            if beam['past'] is None:
                out = model(
                    input_ids=beam['ids'],
                    attention_mask=beam['mask'],
                    use_cache=True
                )
            else:
                out = model(
                    input_ids=beam['ids'][:, -1:],
                    attention_mask=beam['mask'],
                    past_key_values=beam['past'],
                    use_cache=True
                )
            logits = out.logits[:, -1, :]
            past = out.past_key_values
            # calculate the log probability of the generated token
            log_probs = torch.log_softmax(logits, dim=-1)

            topk_logprobs, topk_idx = torch.topk(log_probs, beam_size, dim=-1)
            topk_logprobs = topk_logprobs[0]
            topk_idx = topk_idx[0]
            for j in range(beam_size):
                next_tok = topk_idx[j].unsqueeze(0).unsqueeze(0)
                new_ids = torch.cat([beam['ids'], next_tok], dim=-1)
                new_mask = torch.cat([
                    beam['mask'],
                    torch.ones((1,1), dtype=beam['mask'].dtype, device=device)
                ], dim=-1)
                new_score = beam['score'] + topk_logprobs[j].item()
                done = (next_tok.item() == eos_id)
                all_candidates.append({
                    'ids': new_ids,
                    'mask': new_mask,
                    'past': past,
                    'score': new_score,
                    'done': done
                })
        # select top beams
        beams = sorted(all_candidates, key=lambda x: x['score'], reverse=True)[:beam_size]
        if all(b['done'] for b in beams):
            break
    # pick best beam
    best_beam = max(beams, key=lambda x: x['score'])
    return best_beam['ids'][:, seq_len:]

def best_of_n(model, input_ids, attention_mask, args):
    """
    Run Best-of-N sampling: draw args.best_of samples via multinomial sampling
    and pick the sequence with the highest sum of log-probabilities.
    """
    eos_id = model.config.eos_token_id
    device = input_ids.device
    seq_len = input_ids.size(1)
    best_seq = None
    best_score = float('-inf')
    for _ in range(args.best_of):
        # sampling pass
        generated = input_ids
        past = None
        mask = attention_mask
        # generate new tokens till EOS or max_new_tokens is reached
        for _ in range(args.max_new_tokens):
            if past is None:
                out = model(
                    input_ids=generated,
                    attention_mask=mask,
                    use_cache=True
                )
            else:
                out = model(
                    input_ids=generated[:, -1:],
                    attention_mask=mask,
                    past_key_values=past,
                    use_cache=True
                )
            logits = out.logits[:, -1, :] # batch, last token is the predicted token, logits over vocabulary
            past = out.past_key_values
            # apply temperature
            logits = logits / args.temperature
            # top-k if requested
            if args.k > 0:
                # top k returns the values and the indices across the batch
                topk_vals, topk_idx = torch.topk(logits, args.k, dim=-1) # gets the top k probability tokens
                # create a probability distribution over the vocabulary
                # zeros_like creates a tensor with zeros with the same shape as logits
                # scatter_ takes the softmax of logits and puts value in correct position using topk_idx along the first dimension
                # final dimension should be batch, probability distribution over vocabulary
                # the second dimension is full vocab but only top k slots are non-zero
                probs = torch.zeros_like(logits).scatter_(1, topk_idx, torch.softmax(topk_vals, dim=-1))
            else:
                # probability distribution over the vocabulary
                # final dimension should be batch, probability distribution over vocabulary
                probs = torch.softmax(logits, dim=-1)
            # sample from the probability distribution
            # num_samples=1 means we sample one token from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            # append the sampled token to the existing sequence
            generated = torch.cat([generated, next_token], dim=-1)
            # append the mask to the existing mask
            mask = torch.cat([mask, torch.ones((generated.size(0), 1), dtype=mask.dtype, device=device)], dim=-1)
            # if the sampled token is the end of sentence token, break
            if next_token.item() == eos_id:
                break
        # score this sequence
        with torch.no_grad():
            full_out = model(input_ids=generated, attention_mask=mask, use_cache=False)
            log_probs = torch.log_softmax(full_out.logits, dim=-1)
        score = 0.0
        for i in range(seq_len, generated.size(1)):
            token_id = generated[0, i]
            score += log_probs[0, i-1, token_id].item()
        if score > best_score:
            best_score = score
            best_seq = generated[:, seq_len:]
    return best_seq

def generate(model, input_ids, attention_mask, args):
    """
    Route to a generation strategy. Uses model.generate for most, manual best_of_n for 'best_of'.
    """
    if args.strategy == 'greedy':
        return greedy_search(model, input_ids, attention_mask, args)
    elif args.strategy == 'beam_search':
        return beam_search(model, input_ids, attention_mask, args)
    elif args.strategy == 'best_of':
        return best_of_n(model, input_ids, attention_mask, args)
    elif args.strategy == 'top_k':
        return top_k(model, input_ids, attention_mask, args)
    elif args.strategy == 'top_p':
        return top_p(model, input_ids, attention_mask, args)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='distilbert/distilgpt2', help='HuggingFace model')
    parser.add_argument('--strategy', choices=[
        'greedy','beam_search','best_of','top_k','top_p'
    ], default='best_of')
    parser.add_argument('--num_beams', type=int, default=2, help='Number of beams')
    parser.add_argument('--best_of', type=int, default=2, help='Best-of-N samples')
    parser.add_argument('--max_new_tokens', type=int, default=10, help='Max tokens to generate')
    parser.add_argument('--num_samples', type=int, default=1, help='Examples to evaluate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    parser.add_argument('--k', type=int, default=5, help='Top-k sampling')
    parser.add_argument('--p', type=float, default=0.9, help='Top-p sampling')
    args = parser.parse_args()

    if args.verbose:
        print(f"Args: {args}")

    # load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)

    # load dataset
    print("Loading MMLU validation split...")
    ds = load_dataset("cais/mmlu", 'all', split="validation")
    ds = ds.select(range(min(args.num_samples, len(ds))))
    print(f"Evaluating {len(ds)} examples with strategy '{args.strategy}'")

    prompts, preds, trues = [], [], []
    for ex in tqdm(ds, desc="Generation"):
        question, choices, label = ex['question'], ex.get('choices', []), ex['answer']
        choices_str = ' '.join(f"({chr(65+i)}) {c}" for i,c in enumerate(choices))
        prompt = (
            f"You are an assistant. Read the question and answer with a single letter."
            f"\nQuestion: {question}\nChoices: {choices_str}\nAnswer:"
        )
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(args.device) for k,v in inputs.items()}
        seq = generate(model, inputs['input_ids'], inputs['attention_mask'], args)
        text = tokenizer.decode(seq[0], skip_special_tokens=True).strip()
        pred = text.split()[0].strip("()") if text.split() else ""
        true = chr(65 + int(label))
        prompts.append(prompt)
        preds.append(pred)
        trues.append(true)

    df = pd.DataFrame({
        'prompt': prompts,
        'prediction': preds,
        'true': trues
    })
    out_file = f"generation_{args.model.replace('/','_')}_{args.strategy}.csv"
    df.to_csv(out_file, index=False)
    print(f"Results saved to {out_file}")


if __name__ == '__main__':
    main()
