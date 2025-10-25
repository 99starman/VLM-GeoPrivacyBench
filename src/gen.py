"""Adapted from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference"""
import logging
import os
from argparse import Namespace
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple, Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer  # type: ignore
from vllm import EngineArgs, LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser

from prompts import QUESTION_DATA
from utils import image_to_base64, parse_answers, prepare_question_prompt

MODEL_MAP = {
    "deepseek-vl2": "deepseek-ai/deepseek-vl2",
    "deepseek-vl2-tiny": "deepseek-ai/deepseek-vl2-tiny",
    "deepseek-vl2-small": "deepseek-ai/deepseek-vl2-small",
    "Llama-3.2-11B-Vision-Instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Llama-3.2-90B-Vision-Instruct": "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-32B-Instruct": "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-72B-Instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "InternVL2_5-38B": "OpenGVLab/InternVL2_5-38B"
}

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompt: str
    stop_token_ids: Optional[List[int]] = None
    chat_template: Optional[str] = None
    lora_requests: Optional[List[LoRARequest]] = None

def run_deepseek_vl2(version: str, sys_msg: str, usr_msgs: List[str]) -> ModelRequestData:
    model_name = MODEL_MAP[version]
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=2,
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        limit_mm_per_prompt={"image": 1},
    )
    prompt = (
        f"<|System|>:{sys_msg}\n\n<|User|>: <image>\n" +
        "".join([f"<|User|>:{usr_msg}" for usr_msg in usr_msgs]) + 
        "\n\n<|Assistant|>:"
    )
    return ModelRequestData(engine_args=engine_args, prompt=prompt)

def run_gemma3(version: str, sys_msg: str, usr_msgs: List[str]) -> ModelRequestData:
    model_name = MODEL_MAP[version]
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 1},
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_msg}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "".join(usr_msgs)}]},
    ]
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return ModelRequestData(engine_args=engine_args, prompt=prompt)

def run_mllama(version: str, sys_msg: str, usr_msgs: List[str]) -> ModelRequestData:
    model_name = MODEL_MAP[version]
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=8192,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 1},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_msg}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "".join(usr_msgs)}]}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return ModelRequestData(engine_args=engine_args, prompt=prompt)

def run_qwen2_5_vl(version: str, sys_msg: str, usr_msgs: List[str]) -> ModelRequestData:
    model_name = MODEL_MAP[version]
    engine_args = EngineArgs(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={"min_pixels": 28*28, "max_pixels": 1280*28*28, "fps": 1},
        limit_mm_per_prompt={"image": 1},
    )
    prompt = (
        f"<|im_start|>system\n{sys_msg}<|im_end|>\n" +
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{''.join(usr_msgs)}<|im_end|>\n" +
        "<|im_start|>assistant\n"
    )
    return ModelRequestData(engine_args=engine_args, prompt=prompt)

def run_internvl2_5(version: str, sys_msg: str, usr_msgs: List[str]) -> ModelRequestData:
    model_name = MODEL_MAP[version]
    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    messages = [
        {
            'role': 'system',
            'content': sys_msg
        },
        {
            'role': 'user',
            'content': f'<image>\n{"".join(usr_msgs)}'
        }
    ]
    prompt = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    return ModelRequestData(
        engine_args=engine_args,
        prompt=prompt,
        stop_token_ids=stop_token_ids,
    )


model_example_map = {
    "deepseek-vl2": run_deepseek_vl2,
    "deepseek-vl2-tiny": run_deepseek_vl2,
    "deepseek-vl2-small": run_deepseek_vl2,
    "Llama-3.2-11B-Vision-Instruct": run_mllama,
    "Llama-3.2-90B-Vision-Instruct": run_mllama,
    "Qwen2.5-VL-3B-Instruct": run_qwen2_5_vl,
    "Qwen2.5-VL-7B-Instruct": run_qwen2_5_vl,
    "Qwen2.5-VL-32B-Instruct": run_qwen2_5_vl,
    "Qwen2.5-VL-72B-Instruct": run_qwen2_5_vl,
    "gemma-3-4b-it": run_gemma3,
    "InternVL2_5-38B": run_internvl2_5,
}



def main(args: Namespace):
    timestamp = datetime.now().strftime("%m-%d-%y-%H:%M:%S")
    log_dir = Path("evaluation/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / f"gen_{timestamp}.log",
        filemode="w",
        format="%(asctime)s.%(msecs)03d-%(name)s-%(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )

    load_dotenv(".env")
    access_token = os.getenv("HF_TOKEN")
    if not access_token:
        raise ValueError("HF_TOKEN not found in .env")
    api_key = os.getenv("AZURE_API_KEY")
    api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    sys_prompt, usr_prompts = prepare_question_prompt(
        args.prompting_method, args.free_form, args.include_heuristics, enforce_format=True
    )

    # Support multiple image directories to reuse one model initialization
    image_dirs = [Path(p) for p in (args.image_dirs or [])]
    if not image_dirs:
        raise ValueError("--image-dirs must be provided (one or more paths)")

    if args.model_type not in model_example_map:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    req_data = model_example_map[args.model_type](args.model_type, sys_prompt, usr_prompts)
    logging.info(f"Model request data:\n{req_data} #####")

    engine_args = asdict(req_data.engine_args)
    engine_args.update(seed=args.seed, tensor_parallel_size=args.num_gpus)
    llm = LLM(**engine_args)

    if req_data.lora_requests:
        for lora in req_data.lora_requests:
            llm.llm_engine.add_lora(lora_request=lora)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        max_tokens=512,
        stop_token_ids=req_data.stop_token_ids
    )
    
    forced_free_form = args.prompting_method in ["iter-cot", "malicious"]
    question_columns = [f"Q{i+1}" for i in range(len(QUESTION_DATA))] if not (args.free_form or forced_free_form) else ["Q7-gen", "Q7-label"]

    def process_one_dir(image_dir_path: Path):
        image_paths = list(image_dir_path.glob("*"))
        if not image_paths:
            raise ValueError(f"No images found in {image_dir_path}")
        df = pd.DataFrame(columns=["id"] + question_columns)  # type: ignore
        for start_idx in tqdm(range(0, len(image_paths), args.batch_size), desc=f"Batch inference {image_dir_path.name} ..."):
            end_idx = min(start_idx + args.batch_size, len(image_paths))
            current_batch_paths = image_paths[start_idx:end_idx]

            batch_inputs = []
            for img_path in current_batch_paths:
                image_data = fetch_image(f"data:image/jpg;base64,{image_to_base64(img_path)}")
                batch_inputs.append({
                    "prompt": req_data.prompt,
                    "multi_modal_data": {"image": image_data}
                })

            if args.vllm_inference_method == "generate":
                outputs = llm.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
            else:
                outputs = []
                for i, inp in enumerate(tqdm(batch_inputs)):
                    output = llm.chat(
                        [
                            {"role": "user", "content": inp["prompt"]},
                            {"role": "user", "content": [
                                {"type": "text", "text": inp["prompt"]},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{image_to_base64(current_batch_paths[i])}"}}
                            ]},
                        ],
                        sampling_params=sampling_params,
                        chat_template=req_data.chat_template,
                        lora_request=req_data.lora_requests, # type: ignore
                        use_tqdm=False
                    )
                    outputs.append(output[0])

            for img_path, output in zip(current_batch_paths, outputs):
                record = {"id": str(img_path.stem)}
                generated = output.outputs[0].text.strip()
                logging.debug(f"####### Generated output #######\n{generated}\n##############")

                answers = parse_answers(generated, (args.free_form or forced_free_form), api_key, api_endpoint)
                if (args.free_form or forced_free_form):
                    record["Q7-gen"] = answers[0]
                    record["Q7-label"] = answers[1]
                else:
                    for i, a in enumerate(answers):
                        record[f"Q{i+1}"] = a.strip()
                df.loc[len(df)] = record

        task_name = args.task_name if args.task_name else image_dir_path.name
        mode = f"{args.vllm_inference_method}_{args.prompting_method}"
        if args.include_heuristics:
            mode += "_heuristics"
        if (args.free_form or forced_free_form):
            mode += "_free-form"

        method_folder = args.prompting_method
        if method_folder in ["iter", "cot"]:
            method_folder = "iter-cot"

        if not args.free_form:
            dest_dir = Path(args.out_dir) / "mcq"
        else:
            dest_dir = Path(args.out_dir) / method_folder / "responses"

        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / f"{mode}_{args.model_type}_{task_name}.json"
        df.to_json(out_path, orient="records")
        logging.info(f"Results saved to {out_path}")

    # Process all requested directories with a single initialized model
    for dir_path in image_dirs:
        process_one_dir(dir_path)



if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument('--model-type',
                        '-m',
                        type=str,
                        choices=MODEL_MAP.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument("--vllm-inference-method",
                        type=str,
                        default="generate",
                        choices=["generate", "chat"],
                        help="The method to run in `vllm.LLM`.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for inference.")
    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="Set the seed when initializing `vllm.LLM`.")
    parser.add_argument("--num-gpus",
                        type=int,
                        default=4,
                        help="Number of GPUs when initializing `vllm.LLM`.")
    parser.add_argument("--image-dirs", nargs='+', type=str, help="One or more image directories to process with one model load.")
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Task name for output path (defaults to image directory name). Use 'combined' for combined dataset.",
    )
    parser.add_argument("--prompting-method",
                        type=str,
                        default="zs",
                        choices=["zs", "iter-cot", "malicious"],
                        help="The method for prompting the model.")
    parser.add_argument(
        "--include-heuristics",
        action="store_true",
        help="whether to include heuristics in the prompt.",
    )
    parser.add_argument(
        "--free-form",
        action="store_true",
        help="whether to prompt for free-form answers.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="evaluation/main",
        help="Directory of the output result.",
    )
    args = parser.parse_args()
    main(args)