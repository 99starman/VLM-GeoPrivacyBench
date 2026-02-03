import argparse
import asyncio
import base64
import datetime
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import anthropic
import openai
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from google import genai as google_genai
from tqdm import tqdm

from prompts import QUESTION_DATA
from utils import (
    rescale_image,
    prepare_question_prompt,
    parse_answers,
    call_api,
    call_api_iterative,
)

MODEL_TYPES = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "o3",
    "o4-mini",
    "gpt-4o",
    "gpt-5",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "gemini-2.5-flash",
    "claude-sonnet-4-20250514",
    "grok-4-fast-reasoning",
]


def process_single_thread(
    prompting_method,
    client,
    model_type,
    sys_prompt,
    usr_prompt,
    image_path,
    figstep_image_path,
    granularity_client,
    seed=None,
    temperature: float = 0.7,
):
    jailbreak_aid_image_path = figstep_image_path if prompting_method == "malicious" else None 
    # Models with very low rate limits - sleep 2 seconds after each API call (vanilla prompting only)
    low_rate_limit_models = ["o3", "Llama-4-Maverick-17B-128E-Instruct-FP8"]
    needs_sleep = any(model in model_type for model in low_rate_limit_models)
    
    if "iter" not in prompting_method:
        res = call_api(
            client,
            model_type,
            sys_prompt,
            usr_prompt,
            image_path,
            jailbreak_aid_image_path,
            seed=seed,
            temperature=temperature,
        )
        if needs_sleep:
            time.sleep(2.0)
        return res
    else:
        # No sleep for iterative CoT calls
        return call_api_iterative(
            client,
            model_type,
            sys_prompt,
            usr_prompt,
            image_path,
            granularity_client=granularity_client,
            seed=seed,
            temperature=temperature,
        )


def main(args):
    timestamp = datetime.datetime.now().strftime("%m-%d-%y-%H:%M:%S")
    log_dir = Path("evaluation/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / f"api_gen_{timestamp}.log",
        filemode="w",
        format="%(asctime)s.%(msecs)03d-%(name)s-%(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO # logging.DEBUG,
    )

    env_file = os.getenv("DOTENV_PATH", ".env")
    logging.info(f"Loading environment variables from: {env_file}")
    load_dotenv(env_file, override=True)
    # Log the endpoint being used
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "not set")
    if azure_endpoint != "not set":
        logging.info(f"Using Azure OpenAI endpoint: {azure_endpoint}")
 
    # Load keys
    azure_api_key: Optional[str] = os.getenv("AZURE_API_KEY")
    azure_openai_endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    oai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    claude_api_key: Optional[str] = os.getenv("CLAUDE_API_KEY")

    if args.use_azure:
        if not azure_api_key or not azure_openai_endpoint:
            raise ValueError("Azure API key and endpoint must be set in the .env file when using Azure")
        if args.model_type == "Llama-4-Maverick-17B-128E-Instruct-FP8":
            azure_inference_sdk_api_endpoint = os.getenv("AZURE_INFERENCE_SDK_ENDPOINT")
            if not azure_inference_sdk_api_endpoint:
                raise ValueError("Azure Inference SDK API endpoint must be set in the .env file when not using Azure OpenAI")
    else:
        if not oai_api_key:
            logging.warning("OPENAI_API_KEY not set; OpenAI direct mode will be unavailable.")

    is_free_form = (args.prompting_method in ["iter-cot", "malicious"]) or args.free_form
    sys_prompt, usr_prompts_list = prepare_question_prompt(
        args.prompting_method, is_free_form, args.include_heuristics, enforce_format=False, q7_only=args.q7_only
    )
    usr_prompt = "\n".join(usr_prompts_list)
    # logging.debug(f"User prompt: {usr_prompt}")

    image_paths = list(Path(args.image_dir).glob("*"))
    
    # Exclude images that exist in the exclude directory
    if args.exclude_dir:
        exclude_dir = Path(args.exclude_dir)
        if exclude_dir.exists() and exclude_dir.is_dir():
            exclude_images = {img.name for img in exclude_dir.glob("*") if img.is_file()}
            original_count = len(image_paths)
            image_paths = [img for img in image_paths if img.name not in exclude_images]
            excluded_count = original_count - len(image_paths)
            if excluded_count > 0:
                logging.info(f"Excluded {excluded_count} images from {args.exclude_dir} (already evaluated)")
        else:
            logging.warning(f"Exclude directory does not exist or is not a directory: {args.exclude_dir}")
    
    if args.max_examples and args.max_examples > 0:
        image_paths = image_paths[: args.max_examples]
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")
    task_name = args.task_name if args.task_name else Path(args.image_dir).name

    if is_free_form:
        question_columns = ["Q7-gen", "Q7-label"]
    elif args.q7_only:
        question_columns = ["Q7"]
    else:
        question_columns = [f"Q{i+1}" for i in range(len(QUESTION_DATA))]

    def _azure_openai_client():
        return openai.AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_api_key,
            api_version="2025-01-01-preview",
        )

    def _require_azure_for_granularity():
        if is_free_form and (not azure_api_key or not azure_openai_endpoint):
            raise ValueError("AZURE_API_KEY and AZURE_OPENAI_ENDPOINT must be set in .env to run granularity labeling with gpt-4o-mini.")

    if "gemini" in args.model_type:
        client = google_genai.Client()
        _require_azure_for_granularity()
        granularity_client = _azure_openai_client()
    elif "claude" in args.model_type:
        client = anthropic.Anthropic(api_key=claude_api_key)
        _require_azure_for_granularity()
        granularity_client = _azure_openai_client()
    elif args.use_azure:
        if args.model_type == "Llama-4-Maverick-17B-128E-Instruct-FP8":
            client = ChatCompletionsClient(
                endpoint=azure_inference_sdk_api_endpoint,
                credential=AzureKeyCredential(azure_api_key),
            )
            granularity_client = _azure_openai_client()
        else:
            client = openai.AzureOpenAI(
                azure_endpoint=azure_openai_endpoint,
                api_key=azure_api_key,
                api_version="2025-01-01-preview",
            )
            granularity_client = client
    else:
        client = openai.OpenAI(api_key=oai_api_key)
        _require_azure_for_granularity()
        granularity_client = _azure_openai_client()

    results = []
    if args.claude_batch and "claude" in args.model_type:
        if not isinstance(client, anthropic.Anthropic):
            client = anthropic.Anthropic(api_key=claude_api_key)

        # Support chunked batches to avoid 413 Payload Too Large
        # https://github.com/anthropics/claude-code/issues/3335
        batch_size = getattr(args, "claude_batch_size", 50)
        custom_id_to_text: dict[str, str] = {}
        for start_idx in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[start_idx:start_idx + batch_size]

            batch_requests = []
            for image_path in chunk_paths:
                image_id = Path(image_path).stem
                # https://github.com/anthropics/claude-code/issues/5419
                image_b64 = rescale_image(Path(image_path), max_bytes=5 * 1024 * 1024)
                content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        }
                    }
                ]
                # In malicious mode, include the jailbreak aid image as a second image
                if args.prompting_method == "malicious" and args.figstep_image_path:
                    try:
                        aid_b64 = rescale_image(Path(args.figstep_image_path), max_bytes=5 * 1024 * 1024)
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": aid_b64,
                            }
                        })
                    except Exception as e:
                        logging.warning(f"Failed to attach jailbreak aid image: {e}")
                content.append({"type": "text", "text": usr_prompt})
                params = {
                    "model": args.model_type,
                    "max_tokens": 1200,
                    "system": sys_prompt,
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                    "messages": [{"role": "user", "content": content}],
                }
                # Claude extended thinking doesn't support non-default temperature; only include it when set to 1.0.
                if args.temperature == 1.0:
                    params["temperature"] = args.temperature
                # Claude batch API does not support seed parameter
                # Skip seed for Claude models to maintain original behavior
                if args.seed is not None and "claude" not in args.model_type.lower():
                    params["seed"] = args.seed
                batch_requests.append({"custom_id": image_id, "params": params})

            message_batch = client.beta.messages.batches.create(requests=batch_requests)
            batch_id = getattr(message_batch, "id", None)
            logging.info(f"Created Claude batch: {batch_id} for items {start_idx}-{start_idx + len(chunk_paths) - 1}")

            # Poll with timeout
            max_wait_time = 600
            poll_interval = 5
            elapsed_time = 0
            while elapsed_time < max_wait_time:
                batch_info = client.beta.messages.batches.retrieve(batch_id)
                status = getattr(batch_info, "processing_status", None)
                
                # Log full batch info for debugging
                logging.debug(f"Full batch info: {batch_info}")
                request_counts = getattr(batch_info, "request_counts", None)
                if request_counts:
                    logging.info(f"Claude batch {batch_id} status: {status}, counts: {request_counts} (elapsed: {elapsed_time}s)")
                else:
                    logging.info(f"Claude batch {batch_id} status: {status} (elapsed: {elapsed_time}s)")
                
                # Check if batch is complete - status should be "ended" and all requests processed
                if status == "ended":
                    break
                # Also check if all requests are done (succeeded + errored + expired = total)
                if request_counts:
                    total = getattr(request_counts, "processing", 0) + getattr(request_counts, "succeeded", 0) + getattr(request_counts, "errored", 0) + getattr(request_counts, "expired", 0) + getattr(request_counts, "canceled", 0)
                    completed = getattr(request_counts, "succeeded", 0) + getattr(request_counts, "errored", 0) + getattr(request_counts, "expired", 0)
                    if total > 0 and completed == total:
                        logging.info(f"All {total} requests completed (succeeded={getattr(request_counts, 'succeeded', 0)}, errored={getattr(request_counts, 'errored', 0)}, expired={getattr(request_counts, 'expired', 0)})")
                        break
                
                time.sleep(poll_interval)
                elapsed_time += poll_interval
            
            if elapsed_time >= max_wait_time:
                logging.error(f"Claude batch {batch_id} timed out after {max_wait_time}s")
                continue  # Skip to next chunk

            # Retrieve results for this chunk
            logging.info(f"Retrieving results for batch {batch_id}")
            results_count = 0
            try:
                for result in client.beta.messages.batches.results(batch_id):
                    results_count += 1
                    res_type = getattr(result.result, "type", None)
                    custom_id = str(getattr(result, "custom_id", ""))
                    logging.debug(f"Processing result {results_count} with custom_id: {custom_id}, type: {res_type}")
                    
                    if res_type == "succeeded":
                        message = getattr(result.result, "message", None)
                        usage = getattr(message, "usage", None) if message is not None else None
                        if usage is not None:
                            try:
                                usage_json = json.loads(json.dumps(usage, default=lambda o: getattr(o, "__dict__", str(o))))
                                logging.info(f"Claude batch usage for {custom_id}: {json.dumps(usage_json)}")
                            except Exception:
                                logging.info(f"Claude batch usage (raw) for {custom_id}: {usage}")
                        content = getattr(message, "content", []) if message is not None else []
                        text_parts = []
                        for p in content or []:
                            if getattr(p, "type", None) == "text":
                                text_parts.append(getattr(p, "text", ""))
                        custom_id_to_text[custom_id] = "".join(text_parts).strip()
                    elif res_type == "errored":
                        logging.error(f"Batch item errored: {result}")
                    elif res_type == "expired":
                        logging.error(f"Batch item expired: {result}")
                
                logging.info(f"Retrieved {results_count} results from batch {batch_id}")
            except Exception as e:
                logging.error(f"Error retrieving batch results for {batch_id}: {e}")
                logging.error(f"Exception details: {type(e).__name__}: {str(e)}")

        # Build results using our parsing pipeline
        for image_path in image_paths:
            image_id = Path(image_path).stem
            output = custom_id_to_text.get(image_id, "")
            result_entry = {"id": image_id}
            if is_free_form:
                parsed = parse_answers(
                    output,
                    free_form=True,
                    api_key=azure_api_key,
                    api_endpoint=azure_openai_endpoint,
                    client=granularity_client,
                    granularity_judge_model=args.granularity_judge_model,
                )
                result_entry = {
                    "id": image_id,
                    question_columns[0]: parsed[0] if parsed else "",
                    question_columns[1]: parsed[1] if parsed else "N/A",
                }
            else:
                answers = parse_answers(output, free_form=False, q7_only=args.q7_only, client=granularity_client)
                result_entry = {
                    "id": image_id,
                    **{col: answers[i] if i < len(answers) else "N/A" for i, col in enumerate(question_columns)},
                }
            results.append(result_entry)

    else:
        # Default per-item concurrent path
        with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
            futures = {
                executor.submit(
                    process_single_thread,
                    args.prompting_method,
                    client,
                    args.model_type,
                    sys_prompt,
                    usr_prompt,
                    image_path,
                    args.figstep_image_path,
                    granularity_client,
                    args.seed,
                    args.temperature,
                ): image_path
                for image_path in image_paths
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Model {args.model_type} inference {task_name}"
            ):
                image_path = futures[future]
                logging.debug(f"Processing {image_path}")
                try:
                    output = future.result()
                    image_id = Path(image_path).stem

                    result_entry = {"id": image_id}
                    if is_free_form:
                        parsed = parse_answers(
                            output,
                            free_form=True,
                            api_key=azure_api_key,
                            api_endpoint=azure_openai_endpoint,
                            client=granularity_client,
                            granularity_judge_model=args.granularity_judge_model,
                        )
                        result_entry = {
                            "id": image_id,
                            question_columns[0]: parsed[0] if parsed else "",
                            question_columns[1]: parsed[1] if parsed else "N/A",
                        }

                    else:
                        answers = parse_answers(output, free_form=False, q7_only=args.q7_only, client=granularity_client)
                        result_entry = {
                            "id": image_id,
                            **{
                                col: answers[i] if i < len(answers) else "N/A"
                                for i, col in enumerate(question_columns)
                            },
                        }
                    logging.debug(f"Processed {image_path.name}: {result_entry}")
                    results.append(result_entry)

                except Exception as e:
                    logging.error(f"Failed to do inference on {image_path.name}: {e}")

    # Save results
    mode = f"api_gen_{args.prompting_method}"
    if args.include_heuristics:
        mode += "_heuristics"
    if is_free_form:
        mode += "_free-form"
    if args.q7_only and not is_free_form:
        mode += "_q7-only"
    
    # Decide organized destination directory based on mode/flags
    # Map prompting method to folder names
    method_folder = args.prompting_method
    if method_folder in ["iter", "cot"]:
        method_folder = "iter-cot"

    if not is_free_form:
        # MCQ outputs (with or without heuristics) -> mcq
        dest_dir = Path(args.out_dir) / "mcq"
    else:
        # Free-form regular generations -> <method>/responses
        dest_dir = Path(args.out_dir) / method_folder / "responses"

    dest_dir.mkdir(parents=True, exist_ok=True)

    out_path = dest_dir / f"{mode}_{args.model_type}_{task_name}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        choices=MODEL_TYPES,
        required=True,
        help="Name of the closed model to use",
    )
    parser.add_argument("--image-dir", type=str, required=True, help="Directory of the input images.")
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Task name for output path.",
    )
    parser.add_argument(
        "--figstep-image-path",
        type=str,
        default="data/figstep_jailbreak.png",
        help="Path of the input jailbreaking image following FigStep.",
    )
    parser.add_argument(
        "--prompting-method",
        type=str,
        default="zs",
        choices=["zs", "iter-cot", "malicious"],
        help="The method for prompting the model.",
    )
    parser.add_argument("--n-threads", type=int, default=4, help="Number of threads for parallel processing.")
    parser.add_argument(
        "--include-heuristics", action="store_true", help="Whether to include heuristics in the prompt."
    )
    parser.add_argument("--free-form", action="store_true", help="Whether to prompt for free-form answers.")
    parser.add_argument(
        "--out-dir", type=str, default="evaluation/main", help="Output directory."
    )
    parser.add_argument(
        "--q7-only", action="store_true", help="Whether to prompt for only Q7 in MCQ mode."
    )
    parser.add_argument(
        "--use-azure", action="store_true", help="Whether to use Azure OpenAI"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Early stop after processing this many images",
    )
    parser.add_argument(
        "--claude-batch",
        action="store_true",
        help="Use Claude Message Batches API for Claude runs",
    )
    parser.add_argument(
        "--claude-batch-size",
        type=int,
        default=50,
        help="Number of requests per Claude Message Batch submission",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for model inference (for reproducibility)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for supported models (default: 0.7)",
    )
    parser.add_argument(
        "--granularity-judge-model",
        type=str,
        default="gpt-4.1-mini",
        help="Model to use for granularity label mapping (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--exclude-dir",
        type=str,
        default=None,
        help="Directory containing images to exclude from processing (e.g., already-evaluated samples)",
    )
    args = parser.parse_args()
    main(args)
