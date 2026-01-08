import argparse
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from datasets import load_dataset
from prompts.executor import ExecutorPrompt
from prompts.mutator import MutatorPrompt
from prompts.planner import PlannerPrompt
from prompts.verifier import VerifierPrompt

# ============================================================================
# UTILS
# ============================================================================

class ChunkedWriter:
    def __init__(self, output_dir: str, prefix: str, chunk_size: int = 1000):
        self.output_dir = os.path.join(output_dir, prefix)
        self.prefix = prefix
        self.chunk_size = chunk_size
        self.buffer = []
        self.chunk_counter = 0

        os.makedirs(self.output_dir, exist_ok=True)

    def add_batch(self, items: List[dict]):
        self.buffer.extend(items)
        while len(self.buffer) >= self.chunk_size:
            chunk = self.buffer[: self.chunk_size]
            self.buffer = self.buffer[self.chunk_size :]
            self.flush_chunk(chunk)

    def flush_chunk(self, chunk):
        if not chunk:
            return

        df = pd.DataFrame(chunk)
        filename = f"part_{self.chunk_counter}.parquet"
        path = os.path.join(self.output_dir, filename)

        df.to_parquet(path, index=False)
        print(f"Saved {path} ({len(df)} rows)")

        self.chunk_counter += 1

    def close(self):
        if self.buffer:
            self.flush_chunk(self.buffer)
            self.buffer = []


def remove_think_tags(text: str) -> str:
    """Removes <think>...</think> block from text"""
    if text is None:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()


def apply_template(
    tokenizer, user: str, system: str | None = None, enable_thinking: bool = True
) -> str:
    messages = [{"role": "user", "content": user}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

# ============================================================================
# LOADING UTILS
# ============================================================================

def iter_parquet_file(path: str):
    """
    Iterates over a parquet file batch by batch using pyarrow.
    """
    if not PYARROW_AVAILABLE:
        raise ImportError("pyarrow is required for reading large parquet files. pip install pyarrow")
    
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches():
        df = batch.to_pandas()
        for _, row in df.iterrows():
            yield row.to_dict()

# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class TransformersEngine:
    def __init__(self, model_name: str, cache_dir: str = None, load_in_4bit: bool = False):
        print(f"Loading model from: {model_name}")
        
        # Check if local path exists
        if os.path.exists(model_name):
            print(f"  [+] Found local directory: {model_name}")
            if not os.path.isdir(model_name):
                print(f"  [!] Warning: {model_name} exists but is not a directory.")
        else:
            print(f"  [!] Warning: Path {model_name} not found locally. Transformers might try to download it as a repo ID.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Load params
        kwargs = {
            "device_map": "auto",
            "dtype": dtype, 
            "cache_dir": cache_dir,
            "trust_remote_code": True,
        }
        
        # Only add local_files_only if it exists, to prevent validation errors on bad paths
        if os.path.exists(model_name):
            kwargs["local_files_only"] = True
        
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
            
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying again without local_files_only...")
            if "local_files_only" in kwargs:
                del kwargs["local_files_only"]
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        except:
             # Fallback if tokenizer not in same dir (unlikely for local)
             print("Tokenizer load failed, assuming standard.")
             self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Ensure pad token is set for batching
        # Ensure pad token is set for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Always set left padding for decoder-only models (fixes warning)
        self.tokenizer.padding_side = 'left'

    def generate(self, prompts: List[str], sampling_params: Dict[str, Any] = None) -> List[str]:
        if not prompts:
            return []
            
        # Default params
        if sampling_params is None:
            sampling_params = {}
            
        temperature = sampling_params.get("temperature", 0.0) 
        top_p = sampling_params.get("top_p", 1.0)
        max_new_tokens = sampling_params.get("max_tokens", 1024)
        do_sample = temperature > 0
        
        # Prepare Batch
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        # Decode
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = generated_ids[:, input_len:]
        
        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return decoded

# ============================================================================
# PIPELINE STAGES
# ============================================================================

def process_batch(
    engine: TransformersEngine,
    batch: List[Dict[str, Any]],
    tokenizer, # Passed but we might use engine.tokenizer
    sampling_params: Dict[str, Any],
    writers: Dict[str, ChunkedWriter],
):
    if not batch:
        return

    # -------------------------------------------------------------------------
    # STAGE 1: PLANNER
    # -------------------------------------------------------------------------
    prompts_planner = []
    for item in batch:
        p = apply_template(
            tokenizer,
            PlannerPrompt.user.format(
                problem=item["problem"],
                solution=item["generated_solution"],
                answer=item["expected_answer"],
            ),
            system=PlannerPrompt.system,
        )
        prompts_planner.append(p)

    outputs_planner = engine.generate(prompts_planner, sampling_params)

    batch_with_plans = []
    planner_results = []

    for i, plan in enumerate(outputs_planner):
        item = batch[i]
        planner_results.append({
            "problem": item["problem"],
            "assistant_response": plan,
            "expected_answer": item["expected_answer"],
            "problem_source": item["problem_source"],
        })
        batch_with_plans.append(
            {**item, "plan_raw": plan, "plan_clean": remove_think_tags(plan)}
        )

    writers["planner"].add_batch(planner_results)

    # -------------------------------------------------------------------------
    # STAGE 2: EXECUTOR
    # -------------------------------------------------------------------------
    prompts_executor = []
    for item in batch_with_plans:
        p = apply_template(
            tokenizer,
            ExecutorPrompt.user.format(
                problem=item["problem"],
                plan=item["plan_clean"],
                solution=item["generated_solution"],
                answer=item["expected_answer"],
            ),
            system=ExecutorPrompt.system,
        )
        prompts_executor.append(p)

    outputs_executor = engine.generate(prompts_executor, sampling_params)

    batch_with_execs = []
    executor_results = []

    for i, execution in enumerate(outputs_executor):
        item = batch_with_plans[i]
        executor_results.append({
            "problem": item["problem"],
            "plan": item["plan_clean"],
            "assistant_response": execution,
            "expected_answer": item["expected_answer"],
            "problem_source": item["problem_source"],
        })
        batch_with_execs.append(
            {
                **item,
                "execution_raw": execution,
                "execution_clean": remove_think_tags(execution),
            }
        )

    writers["executor"].add_batch(executor_results)

    # -------------------------------------------------------------------------
    # STAGE 3: VERIFIER (ACCEPT)
    # -------------------------------------------------------------------------
    prompts_ver_accept = []
    for item in batch_with_execs:
        p = apply_template(
            tokenizer,
            VerifierPrompt.user_accept.format(
                problem=item["problem"],
                execution=item["execution_clean"],
                answer=item["expected_answer"],
            ),
            system=VerifierPrompt.system,
        )
        prompts_ver_accept.append(p)

    outputs_ver_accept = engine.generate(prompts_ver_accept, sampling_params)

    verifier_results = []
    for i, ver_resp in enumerate(outputs_ver_accept):
        item = batch_with_execs[i]
        verifier_results.append({
            "problem": item["problem"],
            "execution": item["execution_clean"],
            "proposed_answer": item["expected_answer"],
            "correct_answer": item["expected_answer"],
            "assistant_response": ver_resp,
            "label": "accept",
            "problem_source": item["problem_source"],
        })

    # -------------------------------------------------------------------------
    # STAGE 4: MUTATOR (ANSWER)
    # -------------------------------------------------------------------------
    prompts_mut_ans = []
    for item in batch_with_execs:
        p = apply_template(
            tokenizer,
            MutatorPrompt.wrong_answer_user.format(ans=item["expected_answer"]),
            system=None,
        )
        prompts_mut_ans.append(p)

    outputs_mut_ans = engine.generate(prompts_mut_ans, sampling_params)

    batch_with_wrong_ans = []
    for i, wrong_raw in enumerate(outputs_mut_ans):
        wrong_clean = remove_think_tags(wrong_raw).strip()
        batch_with_wrong_ans.append(
            {**batch_with_execs[i], "wrong_answer": wrong_clean}
        )

    # -------------------------------------------------------------------------
    # STAGE 5: MUTATOR (EXECUTION)
    # -------------------------------------------------------------------------
    prompts_mut_exec = []
    for item in batch_with_wrong_ans:
        p = apply_template(
            tokenizer,
            MutatorPrompt.bad_execution_user.format(
                execution=item["execution_clean"], wrong_answer=item["wrong_answer"]
            ),
            system=MutatorPrompt.system,
            enable_thinking=False,
        )
        prompts_mut_exec.append(p)

    outputs_mut_exec = engine.generate(prompts_mut_exec, sampling_params)

    batch_finished = []
    for i, mut_exec_raw in enumerate(outputs_mut_exec):
        mut_exec_clean = remove_think_tags(mut_exec_raw).replace("```", "").strip()
        batch_finished.append(
            {**batch_with_wrong_ans[i], "mutated_execution": mut_exec_clean}
        )

    # -------------------------------------------------------------------------
    # STAGE 6: VERIFIER (REJECT)
    # -------------------------------------------------------------------------
    prompts_ver_reject = []
    for item in batch_finished:
        p = apply_template(
            tokenizer,
            VerifierPrompt.user_reject.format(
                problem=item["problem"],
                execution=item["mutated_execution"],
                wrong_answer=item["wrong_answer"],
                correct_answer=item["expected_answer"],
            ),
            system=VerifierPrompt.system,
        )
        prompts_ver_reject.append(p)

    outputs_ver_reject = engine.generate(prompts_ver_reject, sampling_params)

    for i, ver_resp in enumerate(outputs_ver_reject):
        item = batch_finished[i]
        verifier_results.append({
            "problem": item["problem"],
            "execution": item["mutated_execution"],
            "proposed_answer": item["wrong_answer"],
            "correct_answer": item["expected_answer"],
            "assistant_response": ver_resp,
            "label": "reject",
            "problem_source": item["problem_source"],
        })

    writers["verifier"].add_batch(verifier_results)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate datasets (Transformers)")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_dir", type=str, default="./datasets")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4, help="Keep small for GPUs without vLLM")
    parser.add_argument("--dataset_name", type=str, default="nvidia/OpenMathReasoning")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use bitsandbytes 4bit quantization")

    args = parser.parse_args()

    print(f"Config: {args}")

    # Initialize Engine
    engine = TransformersEngine(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        load_in_4bit=args.load_in_4bit
    )
    
    # Sampling Config
    sampling_params = {
        "temperature": 0.3,
        "max_tokens": 4096,
        "top_p": 0.95
    }

    # Data Source
    print(f"Loading dataset: {args.dataset_name}")
    ds = None
    if os.path.exists(args.dataset_name):
        ext = args.dataset_name.split(".")[-1].lower()
        if ext == "parquet":
             print("Detected parquet file. Using pyarrow iterative loader.")
             ds = iter_parquet_file(args.dataset_name)
        else:
            if ext == "jsonl": ext = "json"
            try:
                ds = load_dataset(ext, data_files=args.dataset_name, split="train", streaming=True)
            except:
                 ds = load_dataset(ext, data_files=args.dataset_name, split="train", streaming=False)
    else:
        ds = load_dataset(args.dataset_name, split="cot", streaming=True, cache_dir=args.cache_dir)

    writers = {
        "planner": ChunkedWriter(args.output_dir, "planner", chunk_size=args.chunk_size),
        "executor": ChunkedWriter(args.output_dir, "executor", chunk_size=args.chunk_size),
        "verifier": ChunkedWriter(args.output_dir, "verifier", chunk_size=args.chunk_size),
    }

    current_batch = []
    total_processed = 0

    try:
        for row in ds:
            if args.num_samples != -1 and total_processed >= args.num_samples:
                break
            
            # Standardization
            current_dict = {
                "problem": row.get("problem", row.get("question", "")),
                "generated_solution": row.get("generated_solution", row.get("solution", "")),
                "expected_answer": row.get("expected_answer", row.get("answer", "")),
                "problem_source": row.get("problem_source", ""),
            }

            current_batch.append(current_dict)

            if len(current_batch) >= args.batch_size:
                print(f"Processing {total_processed + 1}..{total_processed + len(current_batch)}")
                process_batch(engine, current_batch, engine.tokenizer, sampling_params, writers)
                total_processed += len(current_batch)
                current_batch = []

        if current_batch:
            if args.num_samples == -1 or total_processed < args.num_samples:
                process_batch(engine, current_batch, engine.tokenizer, sampling_params, writers)
                total_processed += len(current_batch)

    except KeyboardInterrupt:
        print("\nInterrupted!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        for w in writers.values():
            w.close()
        print(f"DONE! Processed {total_processed}")

if __name__ == "__main__":
    main()
