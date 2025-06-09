import torch
import torch.distributed as dist
import time
from pathlib import Path
import os

from train import create_parser
from args import parse_json_args
from lightning.fabric import seed_everything
from yalis import ModelConfig, InferenceConfig, LLMEngine
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor.lowering")


# for pretty printing
BLUE = '\033[94m'
GREEN = '\033[92m'
ENDC = '\033[0m'

def print_colored_block(text, color, flush=True):
    for line in text.splitlines():
        print(f"{color}{line}{ENDC}", flush=flush)

def print_rank0(msg, flush=True, color=None):
    if dist.get_rank() == 0:
        if color is None:
            print(msg, flush=flush)
        else:
            print_colored_block(msg, color, flush)



if __name__ == "__main__":
    # Parse arguments
    parser = create_parser()
    parser_args = parser.parse_args()
    args = parse_json_args(parser_args.config_file)
    # Create lightning fabric object
    seed_everything(args.seed)

    with open("data/inference/prompts.txt", 'r') as file:
        prompts = [line.strip() for line in file if line.strip()]

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    formatted_prompts = []
    
    for user_prompt in prompts:
        system_prompt = "You are a helpful chatbot. Answer the following question.\n"
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(conversation, 
                add_generation_prompt=True, 
                tokenize=False)
        formatted_prompts.append(formatted_prompt)


    # Model Config
    model_config = ModelConfig(model_name=args.model_id, precision=args.precision)
    inference_config = InferenceConfig(batch_size=len(formatted_prompts), 
                                       max_length_of_generated_sequences=2*args.tokens_to_generate,
                                       top_p=0.80,
                                       temperature=1.0, 
                                       tp_dims=tuple(args.tp_dimensions) if len(args.tp_dimensions) != 0 else None,
                                       attention_backend="flash",
                                       use_paged_kv_caching=False)


    engine = LLMEngine(model_config=model_config, inference_config=inference_config)

    for iter in range(3):
        output_tokens, metrics = engine.generate(
            formatted_prompts, report_throughput=True, tokens_to_generate=args.tokens_to_generate
        )

    output_tokens = output_tokens.cpu()


    # Decode the token IDs into text
    detokenized_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    for prompt, output in zip(prompts, detokenized_text):
        print_rank0(f"-"*40 + "\n")
        print_rank0(f"{BLUE}User: {prompt}")
        print_rank0(f"AI Assistant: {output}", color=GREEN)

    print_rank0(f"-"*40 + "\n")
    print_rank0(f"Throughput: {metrics['Throughput']} tok/s")

