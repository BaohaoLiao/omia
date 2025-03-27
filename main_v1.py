import os
import re
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

import torch
from vllm import LLM, SamplingParams


SYSTEM_PROMPTS = {
    0: {
        "system": "Solve the math problem from the user. Only work with exact numbers. Only submit an answer if you are sure. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}.",
        "suffix": "",
        "thought_prefix": "<think>\nAlright, we have a math problem.\n\nHmm, it seems that I was asked to use exact numbers.\n\nThis means I should not be approximating calculations.\n\nThis means I should use fractions instead of decimals.\n\nThis means I should avoid cumbersome calculations.\n\nAlso, I should not submit answers that I am not sure.\n\nI should not be submitting guesses.",
    },
    1: {
        "system": "Please reason step by step, and put your final answer within \\boxed{}.",
        "suffix": "",
        "thought_prefix": "<think>\n",
    },
    2: {
        "system": "You are a helpful and harmless math assistant. You should think step-by-step and you are good at reverse thinking to recheck your answer and fix all possible mistakes. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}.",
        "suffix": "",
        "thought_prefix": "<think>\n",
    },
    3: {
        "system": "Please carefully read the problem statement first to ensure you fully understand its meaning and key points. Then, solve the problem correctly and completely through deep reasoning. Finally, return the result modulo 1000 and enclose it in \\boxed{} like \"Atfer take the result modulo 1000, final anwer is \\boxed{180}\".",
        "suffix": "",
        "thought_prefix": "<think>\n",
    },
    4: {
        "system": "Solve the following math problem. Put your final answer within \\boxed{}.",
        "suffix": "\nThe answer is a non-negative interger. If the answer is greater than 999, output answer modulo 1000 instead.",
        "thought_prefix": "<think>\n",
    },
    5: {
        "system": "You are the most powerful math expert. Please solve the problems with deep resoning. You are careful and always recheck your conduction. You will never give answer directly until you have enough confidence. You should think step-by-step, and put the final answer within \\boxed{}.",
        "suffix": "\nThe answer is a non-negative interger. If the answer is greater than 999, output answer modulo 1000 instead.",
        "thought_prefix": "<think>\n",
    },
    6: {
        "system": "Solve the math problem from the user, similar to how a human would (first think how would you solve like a human). Only submit an answer if you are sure. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}.",
        "suffix": "",
        "thought_prefix": "<think>\nAlright, we have a math problem.\n\nHmm, it seems that I was asked to solve like a human. What does that mean? I guess I have to think through the problem step by step, similar to how a person would approach it.\n\nThink deeper. Humans work with easier numbers. They not do insane arithmetic. It means that when I have insane calculations to do, I am likely on the wrong track.\n\nWhat else? This also means I should not be working with decimal places. I should avoid decimals.\n\nAlso, I should not submit answers that I am not sure.",
    },
    7: {
        "system": "Please reason step by step, and put your final answer within \\boxed{}.",
        "suffix": "",
        "thought_prefix": "<think>\n",
    },
    8: {
        "system": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000.",
        "suffix": "",
        "thought_prefix": "<think>\n",
    },
    9: {
        "system": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}.",
        "suffix": "",
        "thought_prefix": "<think>\n",
    },
    10: {
        "system": "Please reason step by step, and put your final answer within \\boxed{} after taking modulo 1000.",
        "suffix": "",
        "thought_prefix": "<think>\n",
    },
    11: {
        "system": "Solve the math problem from the user. Only work with exact numbers. Only submit an answer if you are sure. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}.",
        "suffix": "",
        "thought_prefix": "<think>\n",
    },
    12: {
        "system":  "Solve the math problem from the user, similar to how a human would (first think how would you solve like a human). Only submit an answer if you are sure. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}.",
        "suffix": "",
        "thought_prefix": "<think>\n",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--data_name", default="aimo2", type=str)
    parser.add_argument("--model_name_or_path", default="Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--max_model_len", default=16384, type=int)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--min_p", default=0.05, type=float)
    parser.add_argument("--max_tokens_per_call", default=8192, type=int)
    parser.add_argument("--max_num_seqs", default=16, type=int)
    parser.add_argument("--prompt_type", default=0, type=int)
    parser.add_argument("--sample_idx", default="-1", type=str)
    parser.add_argument("--no_system_prompt", default=False, action='store_true')
    args = parser.parse_args()
    # top_p must be 1 when using greedy sampling (vllm)
    args.top_p = (1 if args.temperature == 0 else args.top_p)  
    return args


def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set as {seed}")


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def select_answer(answers, lengths):
    """
    Select the most common answer with preference to answers that took less tokens.
    If there are ties, select the one with the shortest average length.
    """
    freq_dict = defaultdict(int)
    valid_indices = []  # Keep track of indices where answer is not None
    
    for i, answer in enumerate(answers):
        if answer:
            try:
                if int(answer) == float(answer):
                    freq_dict[int(answer)] += 1
                    valid_indices.append(i)
            except:
                pass

    if len(freq_dict) == 0:
        return 210, 0
    
    # Find the maximum frequency
    max_freq = max(freq_dict.values())
    
    # Get all answers with maximum frequency
    most_common = [ans for ans, freq in freq_dict.items() if freq == max_freq]
    if len(most_common) == 1:
        indices = [i for i in valid_indices if int(answers[i]) == most_common[0]]
        avg_length = sum([lengths[i] for i in indices]) / len(indices)
        return most_common[0] % 1000, avg_length
    
    # If there are ties, calculate average length for each answer
    avg_lengths = {}
    for ans in most_common:
        # Get all indices where this answer appears (excluding None values)
        indices = [i for i in valid_indices if int(answers[i]) == ans]
        # Calculate average length
        avg_length = sum([lengths[i] for i in indices]) / len(indices)
        avg_lengths[ans] = avg_length
    
    # Return the answer with minimum average length
    return min(avg_lengths.items(), key=lambda x: x[1])[0] % 1000, min(avg_lengths.items(), key=lambda x: x[1])[1]


def extract_boxed_text(text):
    """Extract answers from within \boxed{} in the text."""
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    
    for match in matches[::-1]:
        match = match.strip()
        if match != "":
            try:
                if int(float(match)) == float(match):
                    return str(int(float(match)))
            except:
                pass
    return ""


def batch_message_filter(responses):
    """Extract answers from a batch of messages."""
    all_answers = []
    for response in responses:
        answer = extract_boxed_text(response)
        all_answers.append(answer)
    return all_answers


def batch_message_generate(llm, prompts, args):
    sampling_params = SamplingParams(
        temperature=args.temperature,
        min_p=args.min_p,
        top_p=args.top_p,
        skip_special_tokens=False,
        max_tokens=args.max_tokens_per_call,
        #seed=777,
    )
    request_outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )
    request_outputs = sorted(request_outputs, key=lambda x: int(x.request_id))
    responses = [output.outputs[0].text for output in request_outputs]
    return responses


def create_starter_messages(question, args, tokenizer):
    if args.no_system_prompt:
        messages = [
                    {
                        "role": "user", 
                        "content": SYSTEM_PROMPTS[args.prompt_type]["system"] + "\n\nQuestion: " + 
                            question + SYSTEM_PROMPTS[args.prompt_type]["suffix"]
                    },
                ]
    else:
        messages = [
                    {"role": "system", "content": SYSTEM_PROMPTS[args.prompt_type]["system"]},
                    {"role": "user", "content": question + SYSTEM_PROMPTS[args.prompt_type]["suffix"]},
                ]
    prompt = tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True
            ) + SYSTEM_PROMPTS[args.prompt_type]["thought_prefix"]
    prompts = []
    for _ in range(args.max_num_seqs):
        prompts.append(prompt)
    return prompts


def predict_for_question(llm, tokenizer, question, args):
    prompts = create_starter_messages(question, args, tokenizer)
    print("Prompt:", prompts[0])

    responses = batch_message_generate(llm, prompts, args)
    extracted_answers = batch_message_filter(responses)
    lengths = [len(tokenizer.encode(response)) for response in responses]
    answer, length = select_answer(extracted_answers, lengths)
    return responses, extracted_answers, lengths, answer, length


def main(llm, tokenizer, args):
    df = pd.read_csv(os.path.join(args.data_dir, args.data_name + ".csv"))
    sample_ids = []
    for idx in args.sample_idx.split(','):
        if int(idx) == -1:
            sample_ids = list(np.arange(len(df)))
            break
        else:
            sample_ids.append(int(idx))
            
    samples = []
    preds = []
    pass1s = []
    start_time = time.time()
    for index, row in df.iterrows():
        if index not in sample_ids:
            continue

        print("-" * 50)
        print("id:", index)

        gt = row["answer"]
        question = row["problem"]
        if "id" not in row.keys():
            idx = index
        else:
            idx = row["id"]

        responses, extracted_answers, lengths, answer, length = predict_for_question(llm, tokenizer, question, args)
        preds.append(str(answer) == str(gt))
        pass1s.append(sum([str(gt) == str(ans) for ans in extracted_answers])/len(extracted_answers))

        samples.append({
            "id": idx,
            "question": question,
            "completion": responses,
            "preds": extracted_answers,
            "lengths": lengths,
            "pred": answer,
            "length": length,
            "gt": gt,
            "score": preds[-1],
            "pass@1_score": pass1s[-1],
        })

        print("Question:", question)
        print("Predictions:", extracted_answers)
        print("Lengths:", lengths)
        print(f"Prediction: {answer} with {length} generated tokens")
        print("Ground Truth:", gt)
        print(f"Accuracy: {preds[-1]}")
        print(f"Pass@1: {pass1s[-1]}")
        print(f"Overall Accuracy: {sum(preds)/len(preds)}")
        print(f"Overall Pass@1: {sum(pass1s)/len(pass1s)}")


    time_use = time.time() - start_time
    result_json = {
        "time_use_in_minutes": f"{int(time_use // 60)}:{int(time_use % 60):02d}",
        "num_samples": len(df),
        "accuracy": sum(preds) / len(preds),
        "pass@1": sum(pass1s) / len(pass1s),
    }

    out_file_prefix = f"prompt{args.prompt_type}system{args.no_system_prompt}_seed{args.seed}_t{args.temperature}topp{args.top_p}minp{args.min_p}_n{args.max_num_seqs}_len{args.max_tokens_per_call}_sample{args.sample_idx}"
    out_file = f"./outputs/{args.output_dir}/{args.data_name}/{out_file_prefix}.jsonl"
    os.makedirs(f"./outputs/{args.output_dir}/{args.data_name}", exist_ok=True)
    save_jsonl(samples, out_file)
    with open(out_file.replace(".jsonl", f"_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    

def setup(args):
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path,
        max_num_seqs=args.max_num_seqs,   
        max_model_len=args.max_model_len, 
        trust_remote_code=True,      
        tensor_parallel_size=len(available_gpus),
        gpu_memory_utilization=0.95,
        seed=args.seed,
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()
    main(llm, tokenizer, args)


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    setup(args)