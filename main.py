import os
import re
import json
import time
import random
import argparse
import pandas as pd
from collections import Counter

from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--data_name", default="aimo2", type=str)
    parser.add_argument("--model_name_or_path", default="Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--draft_model_name_or_path", default="Qwen2.5-Math-1.5B-Instruct", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_tokens_per_call", default=8192, type=int)
    parser.add_argument("--max_num_seqs", default=16, type=int) 
    parser.add_argument("--speculative_draft_tensor_parallel_size", default=1, type=int)
    parser.add_argument("--enable_log_stats", action="store_true", default=False)
    parser.add_argument("--num_speculative_tokens", default=5, type=int)
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


def select_answer(answers):
    counter = Counter()
    for answer in answers:
        try:
            if int(answer) == float(answer):
                counter[int(answer)] += 1 + random.random() / 1_000
        except:
            pass
    if not counter:
        return 210
    _, answer = sorted([(v,k) for k,v in counter.items()], reverse=True)[0]
    return answer%1000


def batch_message_generate(llm, tokenizer, list_of_messages, args) -> list[list[dict]]:
    sampling_params = SamplingParams(
        temperature=args.temperature,
        min_p=0.01,
        skip_special_tokens=True,
        max_tokens=args.max_tokens_per_call,
    )

    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in list_of_messages
    ]

    request_output = llm.generate(
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )

    sort_keys_and_list_of_messages = []
    for messages, single_request_output in zip(list_of_messages, request_output):
        messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})
        sort_keys_and_list_of_messages.append(
            (
                len(single_request_output.outputs[0].token_ids),
                messages
            )
        )
    sort_keys_and_list_of_messages.sort(key=lambda sort_key_and_messages: sort_key_and_messages[0])

    print([sort_key for sort_key, _ in sort_keys_and_list_of_messages])
    list_of_messages = [messages for _, messages in sort_keys_and_list_of_messages]
    return list_of_messages


def batch_message_filter(list_of_messages):
    extracted_answers = []
    for messages in list_of_messages:
        answer = extract_boxed_text(messages[-1]['content'])
        extracted_answers.append(answer)
    return extracted_answers


def create_starter_messages(question, index):
    options = []
    for _ in range(13):
        options.append(
            [
                {"role": "system", "content": "You are a the most powerful math expert. Please solve the problems with deep resoning. You are careful and always recheck your conduction. You will never give answer directly until you have enough confidence. You should think step-by-step. Return final answer within \\boxed{}, after taking modulo 1000."},
                {"role": "user", "content": question},
            ]
        )
    for _ in range(2):    
        options.append(
            [
                {"role": "system", "content": "You are a helpful and harmless math assistant. You should think step-by-step and you are good at reverse thinking to recheck your answer and fix all possible mistakes. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{}."},
                {"role": "user", "content": question},
            ],
        )
    options.append(
        [
            {"role": "system", "content": "Please carefully read the problem statement first to ensure you fully understand its meaning and key points. Then, solve the problem correctly and completely through deep reasoning. Finally, return the result modulo 1000 and enclose it in \\boxed{} like \"Atfer take the result modulo 1000, final anwer is \\boxed{180}."},
            {"role": "user", "content": question},
        ],
    )
    return options[index%len(options)]


def predict_for_question(llm, tokenizer, question, args):
    list_of_messages = [create_starter_messages(question, index) for index in range(args.max_num_seqs)]
    list_of_messages = batch_message_generate(llm, tokenizer, list_of_messages, args)
    extracted_answers = batch_message_filter(list_of_messages)
    answer = select_answer(extracted_answers)
    return list_of_messages, extracted_answers, answer


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
    llm = LLM(
        model=args.model_name_or_path,
        speculative_model=args.draft_model_name_or_path,
        max_num_seqs=args.max_num_seqs,   
        max_model_len=args.max_tokens_per_call, 
        trust_remote_code=True,      
        tensor_parallel_size=len(available_gpus),
        speculative_draft_tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        seed=args.seed,
        num_speculative_tokens=args.num_speculative_tokens,
    )
    tokenizer = llm.get_tokenizer()
    main(llm, tokenizer, args)


def main(llm, tokenizer, args):
    df = pd.read_csv(os.path.join(args.data_dir, args.data_name, "reference.csv"))
    if args.num_test_sample != -1:
        assert args.num_test_sample <= len(df)
        df = df[:args.num_test_sample]

    samples = []
    preds = []
    start_time = time.time()
    for index, row in df.iterrows():
        gt = row["answer"]
        question = row["problem"]
        idx = row["id"]

        list_of_messages, extracted_answers, answer = predict_for_question(llm, tokenizer, question, args)
        preds.append(int(answer) == int(gt))

        samples.append({
            "id": idx,
            "question": question,
            "completion": list_of_messages,
            "preds": extracted_answers,
            "pred": answer,
            "gt": gt,
        })

    time_use = time.time() - start_time
    result_json = {
        "time_use_in_minutes": f"{int(time_use // 60)}:{int(time_use % 60):02d}",
        "num_samples": len(df),
        "accuracy": sum(preds) / len(preds),
    }

    out_file_prefix = f"seed{args.seed}_t{args.temperature}_{args.num_test_sample}"
    out_file = f"{args.output_dir}/{args.data_name}/{out_file_prefix}.jsonl"
    os.makedirs(f"{args.output_dir}/{args.data_name}", exist_ok=True)
    save_jsonl(samples, out_file)
    with open(out_file.replace(".jsonl", f"_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    

if __name__ == "__main__":
    args = parse_args()
    setup(args)