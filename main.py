import argparse
import os
import re
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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
        logger.info(f"Saved {path} ({len(df)} rows)")

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

def iter_parquet_file(path: str, batch_size: int = 10_000, columns: List[str] = None):
    """
    Memory-efficient iteration over a parquet file.
    
    Args:
        path: Path to parquet file
        batch_size: Number of rows per batch (controls memory usage)
        columns: Optional list of columns to read (reduces memory if you don't need all)
    """
    if not PYARROW_AVAILABLE:
        raise ImportError("pyarrow is required. pip install pyarrow")
    
    parquet_file = pq.ParquetFile(path)
    
    # Можно указать columns для чтения только нужных столбцов
    iter_kwargs = {"batch_size": batch_size}
    if columns:
        iter_kwargs["columns"] = columns
    
    for batch in parquet_file.iter_batches(**iter_kwargs):
        # Вариант 1: Напрямую из PyArrow (самый быстрый)
        for row in batch.to_pylist():
            yield row
        

# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class vLLMEngine:
    def __init__(self, model_name: str, cache_dir: str = None, max_model_len: int = None, 
                 tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
        logger.info(f"Loading model with vLLM from: {model_name}")
        
        # Load tokenizer for prompt length validation
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}. Using default.")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        
        # vLLM initialization
        vllm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
        }
        
        if cache_dir:
            vllm_kwargs["download_dir"] = cache_dir
            
        if max_model_len:
            vllm_kwargs["max_model_len"] = max_model_len
        
        try:
            self.llm = LLM(**vllm_kwargs)
        except Exception as e:
            logger.error(f"Error loading vLLM model: {e}")
            raise
        
        # Get the actual max model length from vLLM
        self.max_model_len = self.llm.llm_engine.model_config.max_model_len
        logger.info(f"Model max context length: {self.max_model_len}")

    def _validate_prompt_length(self, prompt: str, max_tokens: int = 1024) -> bool:
        """Check if prompt + max_tokens fits within model's context window"""
        try:
            tokens = self.tokenizer.encode(prompt)
            prompt_len = len(tokens)
            total_len = prompt_len + max_tokens
            
            if total_len > self.max_model_len:
                logger.warning(
                    f"Skipping prompt: length {prompt_len} + max_tokens {max_tokens} = {total_len} "
                    f"exceeds max_model_len {self.max_model_len}"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating prompt length: {e}")
            return True  # Allow through if validation fails

    def generate(self, prompts: List[str], sampling_params: Dict[str, Any] = None) -> List[str]:
        if not prompts:
            return []
            
        # Default params
        if sampling_params is None:
            sampling_params = {}
            
        temperature = sampling_params.get("temperature", 0.0) 
        top_p = sampling_params.get("top_p", 1.0)
        max_tokens = sampling_params.get("max_tokens", 1024)
        
        # Filter out prompts that are too long
        valid_indices = []
        valid_prompts = []
        
        for i, prompt in enumerate(prompts):
            if self._validate_prompt_length(prompt, max_tokens):
                valid_indices.append(i)
                valid_prompts.append(prompt)
        
        if len(valid_prompts) < len(prompts):
            logger.warning(f"Filtered {len(prompts) - len(valid_prompts)} prompts due to length")
        
        # Create vLLM sampling params
        vllm_sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        # Generate
        if not valid_prompts:
            logger.warning("No valid prompts to generate!")
            return [""] * len(prompts)
        
        outputs = self.llm.generate(valid_prompts, vllm_sampling_params)
        
        # Extract generated text
        generated_texts = [output.outputs[0].text for output in outputs]
        
        # Reconstruct full results array with empty strings for filtered prompts
        results = []
        valid_idx = 0
        for i in range(len(prompts)):
            if i in valid_indices:
                results.append(generated_texts[valid_idx])
                valid_idx += 1
            else:
                results.append("")  # Empty string for filtered prompts
        
        return results

# ============================================================================
# PIPELINE STAGES
# ============================================================================

def process_batch(
    engine: vLLMEngine,
    batch: List[Dict[str, Any]],
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
            engine.tokenizer,
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
            engine.tokenizer,
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
            engine.tokenizer,
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
            engine.tokenizer,
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
            engine.tokenizer,
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
            engine.tokenizer,
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
    parser = argparse.ArgumentParser(description="Generate datasets (vLLM)")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--output_dir", type=str, default="./datasets")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4, help="Number of samples to process together")
    parser.add_argument("--dataset_name", type=str, default="nvidia/OpenMathReasoning")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max_model_len", type=int, default=None, help="Override model's max context length")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization (0.0-1.0)")

    args = parser.parse_args()

    logger.info(f"Config: {args}")

    # Initialize vLLM Engine
    engine = vLLMEngine(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    
    # Sampling Config
    sampling_params = {
        "temperature": 0.3,
        "max_tokens": 4096,
        "top_p": 0.95
    }

    # Data Source
    logger.info(f"Loading dataset: {args.dataset_name}")
    ds = None
    if os.path.exists(args.dataset_name):
        ext = args.dataset_name.split(".")[-1].lower()
        if ext == "parquet":
             logger.info("Detected parquet file. Using pyarrow iterative loader.")
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
        # Wrap dataset iterator with tqdm if possible, but for streaming/generators, simply iterating is fine.
        # If we have num_samples, we can add a total.
        pbar_total = args.num_samples if args.num_samples != -1 else None
        
        # We wrap the loop. Note that since we batch inside, tqdm will tick per sample.
        # However, we yield one by one from ds.
        
        with tqdm(total=pbar_total, desc="Processing Samples", unit="sample") as pbar:
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
                pbar.update(1)

                if len(current_batch) >= args.batch_size:
                    # logger.info(f"Processing batch of size {len(current_batch)}...") # Too verbose if inside using tqdm
                    process_batch(engine, current_batch, sampling_params, writers)
                    total_processed += len(current_batch)
                    current_batch = []

            # Final batch
            if current_batch:
                if args.num_samples == -1 or total_processed < args.num_samples:
                    process_batch(engine, current_batch, sampling_params, writers)
                    total_processed += len(current_batch)

    except KeyboardInterrupt:
        logger.info("Interrupted!")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        for w in writers.values():
            w.close()
        logger.info(f"DONE! Processed {total_processed}")

if __name__ == "__main__":
    main()
