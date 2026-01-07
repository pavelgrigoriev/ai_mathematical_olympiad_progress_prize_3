import argparse
import os
import re
from typing import Any, Dict, List

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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
    """Убирает <think>...</think> блок из текста"""
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

    # Note: enable_thinking is specific to some tokenizers/models variations
    # We will try to pass it, but if it fails (not supported by method), we fallback.
    # vLLM usually takes the raw prompt string.
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
# PIPELINE STAGES
# ============================================================================


def process_batch(
    llm: LLM,
    batch: List[Dict[str, Any]],
    tokenizer,
    sampling_params: SamplingParams,
    writers: Dict[str, ChunkedWriter],
):
    """
    Executes the multi-stage pipeline for a batch of data using vLLM using ONE generate call per stage.
    """
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

    # Generate plans
    outputs_planner = llm.generate(prompts_planner, sampling_params)

    # Store results & prepare next stage inputs
    batch_with_plans = []
    planner_results = []

    for i, output in enumerate(outputs_planner):
        plan = output.outputs[0].text
        item = batch[i]

        planner_results.append(
            {
                "problem": item["problem"],
                "assistant_response": plan,
                "expected_answer": item["expected_answer"],
                "problem_source": item["problem_source"],
            }
        )

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

    outputs_executor = llm.generate(prompts_executor, sampling_params)

    batch_with_execs = []
    executor_results = []

    for i, output in enumerate(outputs_executor):
        execution = output.outputs[0].text
        item = batch_with_plans[i]

        executor_results.append(
            {
                "problem": item["problem"],
                "plan": item["plan_clean"],
                "assistant_response": execution,
                "expected_answer": item["expected_answer"],
                "problem_source": item["problem_source"],
            }
        )

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

    outputs_ver_accept = llm.generate(prompts_ver_accept, sampling_params)

    verifier_results = []
    for i, output in enumerate(outputs_ver_accept):
        ver_resp = output.outputs[0].text
        item = batch_with_execs[i]

        verifier_results.append(
            {
                "problem": item["problem"],
                "execution": item["execution_clean"],
                "proposed_answer": item["expected_answer"],
                "correct_answer": item["expected_answer"],
                "assistant_response": ver_resp,
                "label": "accept",
                "problem_source": item["problem_source"],
            }
        )

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

    outputs_mut_ans = llm.generate(prompts_mut_ans, sampling_params)

    batch_with_wrong_ans = []
    for i, output in enumerate(outputs_mut_ans):
        wrong_raw = output.outputs[0].text
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

    # Note: Mutator Execution uses system prompt and no thinking usually,
    # but here we use same sampling params for simplicity or user preference.
    # Can adjust if needed.
    outputs_mut_exec = llm.generate(prompts_mut_exec, sampling_params)

    batch_finished = []
    for i, output in enumerate(outputs_mut_exec):
        mut_exec_raw = output.outputs[0].text
        # Cleaning
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

    outputs_ver_reject = llm.generate(prompts_ver_reject, sampling_params)

    for i, output in enumerate(outputs_ver_reject):
        ver_resp = output.outputs[0].text
        item = batch_finished[i]

        verifier_results.append(
            {
                "problem": item["problem"],
                "execution": item["mutated_execution"],
                "proposed_answer": item["wrong_answer"],
                "correct_answer": item["expected_answer"],
                "assistant_response": ver_resp,
                "label": "reject",
                "problem_source": item["problem_source"],
            }
        )

    # Add both accept and reject samples to writer
    writers["verifier"].add_batch(verifier_results)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate datasets for Math Olympiad (vLLM Optimized)"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./cache", help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to process. -1 for all.",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Model identifier"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=1000, help="Rows per parquet file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for pipeline processing"
    )
    parser.add_argument(
        "--tp_size", type=int, default=1, help="Tensor Parallelism size (GPU count)"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization (0.0-1.0)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=32768,
        help="Max model length (tokens)",
    )

    args = parser.parse_args()

    print(f"Config:")
    print(f"  Model: {args.model_name}")
    print(f"  Samples: {args.num_samples if args.num_samples != -1 else 'ALL'}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Output: {args.output_dir}")
    print(f"  GPU Util: {args.gpu_memory_utilization}")

    # Initialize vLLM
    print("\nInitializing vLLM...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tp_size,
        download_dir=args.cache_dir,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,  # Adjust for Kaggle T4
        trust_remote_code=True,
        dtype="half",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    sampling_params = SamplingParams(
        temperature=0.3, max_tokens=4096, stop=["<|im_end|>", "<|endoftext|>"]
    )

    # Data Source
    ds = load_dataset(
        "nvidia/OpenMathReasoning",
        split="cot",
        streaming=True,
        cache_dir=args.cache_dir,
    )

    # Writers
    writers = {
        "planner": ChunkedWriter(
            args.output_dir, "planner", chunk_size=args.chunk_size
        ),
        "executor": ChunkedWriter(
            args.output_dir, "executor", chunk_size=args.chunk_size
        ),
        "verifier": ChunkedWriter(
            args.output_dir, "verifier", chunk_size=args.chunk_size
        ),
    }

    current_batch = []
    total_processed = 0

    try:
        for row in ds:
            if args.num_samples != -1 and total_processed >= args.num_samples:
                break

            current_dict = {
                "problem": row["problem"],
                "generated_solution": row["generated_solution"],
                "expected_answer": row["expected_answer"],
                "problem_source": row.get("problem_source", ""),
            }

            current_batch.append(current_dict)

            if len(current_batch) >= args.batch_size:
                print(
                    f"\nProcessing batch {total_processed+1}..{total_processed+len(current_batch)}"
                )
                process_batch(llm, current_batch, tokenizer, sampling_params, writers)
                total_processed += len(current_batch)
                current_batch = []

        # Process remaining
        if current_batch:
            if args.num_samples == -1 or total_processed < args.num_samples:
                print(f"\nProcessing final batch of {len(current_batch)}")
                process_batch(llm, current_batch, tokenizer, sampling_params, writers)
                total_processed += len(current_batch)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving data...")

    finally:
        for w in writers.values():
            w.close()
        print(f"DONE! Processed {total_processed} samples.")


if __name__ == "__main__":
    main()
