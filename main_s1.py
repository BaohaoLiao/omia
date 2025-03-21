import os
import re
import json
import time
import random
import argparse
import pandas as pd
from collections import Counter, defaultdict

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--data_name", default="aimo2", type=str)
    parser.add_argument("--model_name_or_path", default="Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--draft_model_name_or_path", default=None, type=str)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_tokens_per_call", default=8192, type=int)
    parser.add_argument("--max_num_seqs", default=16, type=int) 
    parser.add_argument("--speculative_draft_tensor_parallel_size", default=1, type=int)
    parser.add_argument("--enable_log_stats", action="store_true", default=False)
    parser.add_argument("--num_speculative_tokens", default=5, type=int)
    parser.add_argument("--max_model_len", default=16384, type=int)
    parser.add_argument("--s1", action='store_true', default=False)
    parser.add_argument("--use_math_verify", action='store_true', default=False)
    parser.add_argument("--prompt_type", default=0, type=int)
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def extract_boxed_text(text, use_math_verify=False):
    if use_math_verify:
        from math_verify import LatexExtractionConfig, parse, verify
        from latex2sympy2_extended import NormalizationConfig
        answer_parsed = parse(
                text,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
        if answer_parsed:
            if len(answer_parsed) >=2:
                return str(answer_parsed[1])
            else:
                return str(answer_parsed[0])
        else:
            return ""
    else:
        pattern = r'oxed{(.*?)}'
        matches = re.findall(pattern, text)
        if not matches:
            return ""
        for match in matches[::-1]:
            if match != "":
                return match
        return ""


def select_answer(answers, lengths):
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
        indices = [i for i in valid_indices if answers[i] == str(most_common[0])]
        avg_length = sum([lengths[i] for i in indices]) / len(indices)
        return most_common[0] % 1000, avg_length
    
    # If there are ties, calculate average length for each answer
    avg_lengths = {}
    for ans in most_common:
        # Get all indices where this answer appears (excluding None values)
        indices = [i for i in valid_indices if answers[i] == str(ans)]
        # Calculate average length
        avg_length = sum([lengths[i] for i in indices]) / len(indices)
        avg_lengths[ans] = avg_length
    
    # Return the answer with minimum average length
    return min(avg_lengths.items(), key=lambda x: x[1])[0] % 1000, min(avg_lengths.items(), key=lambda x: x[1])[1]


def batch_message_filter(list_of_messages, use_math_verify):
    extracted_answers = []
    for messages in list_of_messages:
        answer = extract_boxed_text(messages[-1]['content'], use_math_verify=use_math_verify)
        extracted_answers.append(answer)
    return extracted_answers


def batch_message_generate(llm, tokenizer, list_of_messages, args):
    sampling_params = SamplingParams(
        temperature=args.temperature,
        min_p=0.01,
        top_p=args.top_p,
        skip_special_tokens=False,
        max_tokens=args.max_tokens_per_call,
    )

    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        ) + "<think>\n"
        for messages in list_of_messages
    ]

    print("Prompt:", list_of_texts[0])

    request_output = llm.generate(
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )
    request_output = sorted(request_output, key=lambda x: int(x.request_id))

    list_of_lengths_and_messages = []
    for messages, single_request_output in zip(list_of_messages.copy(), request_output):
        messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})
        list_of_lengths_and_messages.append(
            (
                len(single_request_output.outputs[0].token_ids),
                messages
            )
        )

    if args.s1:
        # Obtain answer
        extracted_answers = batch_message_filter([messages for _, messages in list_of_lengths_and_messages], use_math_verify=args.use_math_verify)
        print("First predictions:", extracted_answers)

        good_responses = [] # list of [index, response, length, answer]
        bad_responses = []
        for idx, (prompt, response, answer) in enumerate(zip(list_of_texts, request_output, extracted_answers)):
            if answer:
                good_responses.append((idx, response.outputs[0].text, list_of_lengths_and_messages[idx][0], answer))
            else:
                #if "</think>" in response.outputs[0].text:
                #bad_responses.append((idx, prompt, response.outputs[0].text + "\n\nThus, the final answer is\n\n", list_of_lengths_and_messages[idx][0]))
                #bad_responses.append((idx, prompt, response.outputs[0].text + "\n\n**Final Answer**\n\n", list_of_lengths_and_messages[idx][0]))
                #else:
                if "7B" in args.model_name_or_path:
                    bad_responses.append((idx, prompt, response.outputs[0].text + "\n</think>\n\n**Final Answer**\n\n", list_of_lengths_and_messages[idx][0]))
                else:
                    bad_responses.append((idx, prompt, response.outputs[0].text + "\n</think>\n\n", list_of_lengths_and_messages[idx][0]))

        # Force to generate an answer
        if bad_responses:
            new_list_of_texts = [prompt + response for _, prompt, response, _ in bad_responses]
            new_sampling_params = SamplingParams(
                temperature=args.temperature,
                min_p=0.01,
                top_p=args.top_p,
                skip_special_tokens=False,
                max_tokens=1024,
            )
            new_request_output = llm.generate(
                prompts=new_list_of_texts,
                sampling_params=new_sampling_params,
            )
            new_request_output = sorted(new_request_output, key=lambda x: int(x.request_id))
            new_list_of_lengths_and_messages = []
            for i, (idx, prompt, prev_response, prev_len) in enumerate(bad_responses):
                new_list_of_lengths_and_messages.append(
                    (
                        prev_len + len(new_request_output[i].outputs[0].token_ids),
                        list_of_messages[idx] + [{'role': 'assistant', 'content': prev_response + new_request_output[i].outputs[0].text}]
                    )
                )

                #print("Original:", [prev_response[-200:]])
                #print("New:", [new_request_output[i].outputs[0].text])

            #print("Before:", [length for length, _ in list_of_lengths_and_messages])

            # merge
            for i, (idx, _, _, _) in enumerate(bad_responses):
                list_of_lengths_and_messages[idx] = new_list_of_lengths_and_messages[i]
            
            #print("After:", [length for length, _ in list_of_lengths_and_messages])

    return list_of_lengths_and_messages

SYSTEMP_PROMPTS = [
    {
        "system": "You are a the most powerful math expert. Please solve the problems with deep resoning. You are careful and always recheck your conduction. You will never give answer directly until you have enough confidence. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000.",
        "suffix": "",
    },
    {
        "system": "Please reason step by step, and put your final answer within \\boxed{}.",
        "suffix": "",
    },
    {
        "system": "You are a helpful and harmless math assistant. You should think step-by-step and you are good at reverse thinking to recheck your answer and fix all possible mistakes. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}.",
        "suffix": "",
    },
    {
        "system": "Please carefully read the problem statement first to ensure you fully understand its meaning and key points. Then, solve the problem correctly and completely through deep reasoning. Finally, return the result modulo 1000 and enclose it in \\boxed{} like \"Atfer take the result modulo 1000, final anwer is \\boxed{180}.",
        "suffix": "",
    },
    {
        "system": "Solve the following math problem. Put your final answer within \\boxed{}.",
        "suffix": "\nThe answer is a non-negative interger. If the answer is greater than 999, output answer modulo 1000 instead."
    }
]


def create_starter_messages(question, args):
    options = []
    for _ in range(args.max_num_seqs):
        options.append(
            [
                {"role": "system", "content": SYSTEMP_PROMPTS[args.prompt_type]["system"]},
                {"role": "user", "content": question + SYSTEMP_PROMPTS[args.prompt_type]["suffix"]},
            ]
        )
    return options


def predict_for_question(llm, tokenizer, question, args):
    list_of_messages = create_starter_messages(question, args)
    list_of_lengths_and_messages = batch_message_generate(llm, tokenizer, list_of_messages, args)
    extracted_answers = batch_message_filter([messages for _, messages in list_of_lengths_and_messages], use_math_verify=args.use_math_verify)
    lengths = [length for length, _ in list_of_lengths_and_messages]
    answer, length = select_answer(extracted_answers, lengths)
    return [messages[-1]["content"] for _, messages in list_of_lengths_and_messages], extracted_answers, lengths, answer, length


def save_jsonl(samples, save_path):
    # ensure path
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print("Saved to", save_path)


def setup(args):
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.draft_model_name_or_path is None:
        llm = LLM(
            model=args.model_name_or_path,
            max_num_seqs=args.max_num_seqs,   
            max_model_len=args.max_model_len, 
            trust_remote_code=True,      
            tensor_parallel_size=len(available_gpus),
            gpu_memory_utilization=0.95,
            seed=args.seed,
            enable_prefix_caching=True,
        )
    else:
        ngram_prompt_lookup_max = None
        if args.draft_model_name_or_path == "[ngram]":
            ngram_prompt_lookup_max = 4

        llm = LLM(
            model=args.model_name_or_path,
            speculative_model=args.draft_model_name_or_path,
            max_num_seqs=args.max_num_seqs,   
            max_model_len=args.max_tokens_per_call, 
            trust_remote_code=True,      
            tensor_parallel_size=len(available_gpus),
            speculative_draft_tensor_parallel_size=args.speculative_draft_tensor_parallel_size,
            gpu_memory_utilization=0.95,
            seed=args.seed,
            num_speculative_tokens=args.num_speculative_tokens,
            disable_log_stats=True,
            ngram_prompt_lookup_max=ngram_prompt_lookup_max,
        )
    tokenizer = llm.get_tokenizer()
    main(llm, tokenizer, args)


def main(llm, tokenizer, args):
    df = pd.read_csv(os.path.join(args.data_dir, args.data_name + ".csv"))
    if args.num_test_sample != -1:
        assert args.num_test_sample <= len(df)
        df = df[:args.num_test_sample]

    samples = []
    preds = []
    pass1s = []
    start_time = time.time()
    for index, row in df.iterrows():
        print("-" * 50)
        print("id:", index)

        gt = row["answer"]
        question = row["problem"]
        idx = row["id"]

        list_of_messages, extracted_answers, lengths, answer, length = predict_for_question(llm, tokenizer, question, args)
        preds.append(int(answer) == int(gt))
        pass1s.append(sum([str(gt) == str(ans) for ans in extracted_answers])/len(extracted_answers))

        samples.append({
            "id": idx,
            "question": question,
            "completion": list_of_messages,
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

    out_file_prefix = f"prompt{args.prompt_type}_seed{args.seed}_t{args.temperature}_n{args.max_num_seqs}_len{args.max_tokens_per_call}_{args.num_test_sample}"
    out_file = f"./outputs/{args.output_dir}/{args.data_name}/{out_file_prefix}.jsonl"
    os.makedirs(f"./outputs/{args.output_dir}/{args.data_name}", exist_ok=True)
    save_jsonl(samples, out_file)
    with open(out_file.replace(".jsonl", f"_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    

if __name__ == "__main__":
    args = parse_args()
    setup(args)