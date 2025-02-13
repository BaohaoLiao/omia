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
    parser.add_argument("--batch_size", default=10, type=int)
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
                return answer_parsed[1]
            else:
                return answer_parsed[0]
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


def batch_message_generate(llm, tokenizer, messages, args):
    prompts = [
        tokenizer.apply_chat_template(
            conversation=message,
            tokenize=False,
            add_generation_prompt=True
        ) + "<think>\n" for message in messages
    ]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        min_p=0.01,
        top_p=args.top_p,
        skip_special_tokens=False,
        max_tokens=args.max_tokens_per_call,
        n=args.max_num_seqs,
    )
    responses = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )
    responses = sorted(responses, key=lambda x: int(x.request_id))

    list_of_lengths_and_messages = [[] for _ in range(len(messages))]
    for i, (message, response) in enumerate(zip(messages, responses)):
        assert args.max_num_seqs == len(response.outputs)
        for j in range(args.max_num_seqs):
            list_of_lengths_and_messages[i].append(
                len(response.outputs[j].token_ids),
                message + [{'role': 'assistant', 'content': response.outputs[j].text}]
            )

    if args.s1: # enforce to generate an answer
        # Obtain answer
        list_of_extracted_answers = []
        for i, lengths_and_messages in enumerate(list_of_lengths_and_messages):
            list_of_extracted_answers.append(
                batch_message_filter([message for _, message in lengths_and_messages], use_math_verify=args.use_math_verify)
            )

        list_of_good_responses = [[] for _ in range(len(messages))] # list of [index, response, length, answer]
        list_of_bad_responses = [[] for _ in range(len(messages))]
        for i, (prompt, response, extracted_answers) in enumerate(zip(prompts, responses, list_of_extracted_answers)):
            for j, answer in enumerate(extracted_answers):
                if answer:
                    list_of_good_responses[i].append((i, j, response.outputs[j].text, list_of_lengths_and_messages[i][j][0], answer))
                else:
                    if "7B" in args.model_name_or_path:
                        list_of_bad_responses[i].append((i, j, prompt, response.outputs[j].text + "\n</think>\n\n**Final Answer**\n\n", list_of_lengths_and_messages[i][j][0]))
                    else:
                        list_of_bad_responses[i].append((i, j, prompt, response.outputs[j].text + "\n</think>\n\n", list_of_lengths_and_messages[i][j][0]))

        # Force to generate an answer
        bad_responses = [bad_response for bad_responses in list_of_bad_responses for bad_response in bad_responses]
        if bad_responses:
            new_prompts = [prompt + response for _, _, prompt, response, _ in bad_responses]
            new_sampling_params = SamplingParams(
                temperature=args.temperature,
                min_p=0.01,
                top_p=args.top_p,
                skip_special_tokens=False,
                max_tokens=1024,
                n=1,
            )
            new_responses = llm.generate(
                prompts=new_prompts,
                sampling_params=new_sampling_params,
            )
            new_responses = sorted(new_responses, key=lambda x: int(x.request_id))

            # merge
            count = 0
            for i, bad_responses in enumerate(list_of_bad_responses):
                if bad_responses:
                    for i, j, prompt, prev_response, prev_len in bad_responses:
                        list_of_lengths_and_messages[i][j] = (
                            prev_len + len(new_responses[count].outputs[0].token_ids),
                            messages[i] + [{'role': 'assistant', 'content': prev_response + new_responses[count].outputs[0].text}]
                        )
                        count += 1

    return list_of_lengths_and_messages


def prepare_prompts(problems):
    messages = []
    for problem in problems:
        message = [
            {"role": "system", "content": "You are a the most powerful math expert. Please solve the problems with deep resoning. You are careful and always recheck your conduction. You will never give answer directly until you have enough confidence. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000."},
            {"role": "user", "content": problem},
        ]
        messages.append(message)
        """
        prompts.append(
            tokenizer.apply_chat_template(
                conversation=message,
                tokenize=False,
                add_generation_prompt=True
            ) + "<think>\n"
        )
        """
    return messages


def predict_for_questions(llm, tokenizer, questions, args):
    messages = messages = prepare_prompts(questions)
    list_of_lengths_and_messages = batch_message_generate(llm, tokenizer, messages, args)

    list_of_extracted_answers = []
    for i, lengths_and_messages in enumerate(list_of_lengths_and_messages):
        list_of_extracted_answers.append(
            batch_message_filter([message for _, message in lengths_and_messages], use_math_verify=args.use_math_verify)
        )

    selected_answers_and_lengths = []
    for extracted_answers, lengths_and_messages in zip(list_of_extracted_answers, list_of_lengths_and_messages):
        lengths = [length for length, _ in lengths_and_messages]
        answer, length = select_answer(extracted_answers, lengths)
        selected_answers_and_lengths.append((answer, length))

    return list_of_lengths_and_messages, list_of_extracted_answers, selected_answers_and_lengths


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

    start_time = time.time()
    problems = [row["problem"] for _, row in df.iterrows()]
    all_samples = []
    for i in range(0, len(problems), args.batch_size):
        batch_problems = problems[i:i + args.batch_size]
        list_of_lengths_and_messages, list_of_extracted_answers, selected_answers_and_lengths = predict_for_questions(llm, tokenizer, batch_problems, args)

        for j in range(len(batch_problems)):
            all_samples.append({
                "id": i + j,
                "question": batch_problems[i],
                "completions": [message[-1]["content"] for length, message in list_of_lengths_and_messages[j]],
                "preds": list_of_extracted_answers[j],
                "lengths": [length for length, message in list_of_lengths_and_messages[j]],
                "pred": selected_answers_and_lengths[j][0],
                "length": selected_answers_and_lengths[j][1],
                "gt": df[i+j]["answer"],
                "score": df[i+j]["answer"] == selected_answers_and_lengths[j][0],
                "pass@1": sum([str(df[i+j]["answer"]) == str(ans) for ans in list_of_extracted_answers[j]])/len(extracted_answers),
            })


    time_use = time.time() - start_time
    result_json = {
        "time_use_in_minutes": f"{int(time_use // 60)}:{int(time_use % 60):02d}",
        "num_samples": len(df),
        "accuracy": sum([sample["score"] for sample in all_samples]) / len(df),
        "pass@1": sum([sample["pass@1"] for sample in all_samples]) / len(df),
    }

    out_file_prefix = f"seed{args.seed}_t{args.temperature}_n{args.max_num_seqs}_len{args.max_tokens_per_call}_{args.num_test_sample}"
    out_file = f"./outputs/{args.output_dir}/{args.data_name}/{out_file_prefix}.jsonl"
    os.makedirs(f"./outputs/{args.output_dir}/{args.data_name}", exist_ok=True)
    save_jsonl(all_samples, out_file)
    with open(out_file.replace(".jsonl", f"_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    

if __name__ == "__main__":
    args = parse_args()
    setup(args)