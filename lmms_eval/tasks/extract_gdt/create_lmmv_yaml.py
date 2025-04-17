#!/usr/bin/env python3
"""
Generate YAML configuration files for lmms-eval tasks
based on different prompt and shot combinations.
Then run each task independently, logging output.
"""
import os
import yaml
import subprocess
import time
import re
from datetime import datetime

# Base directory for the YAML files
BASE_DIR = "/data2/Users/clark/lmms-eval/lmms_eval/tasks/extract_gdt"

# Log directory
LOG_DIR = "/data2/Users/clark/lmms-eval/logs"

# Create directories if they don't exist
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Base template
base_config = """
include: extract_gdt.yaml
dataset_path: "/data2/Users/clark/hadrian_vllm/data/hf_datasets/read_gdt_for_id_from_render-single_images-{prompt}-{n_shot}-{eg_per}-{turn}"
task: "extract_gdt_{n_shot}_{eg_per}_{prompt_suffix}_{turn}"
"""

# Configurations
PROMPTS = [("prompt6_claude_try", ""), ("prompt6_claude_try_answers", "answer")]

SHOTS = [
    # ("1", "50"),
    # ("1", "5"),
    ("3", "50"),
    ("3", "5"),
    # ("9", "5")
    ("9", "50"),
    ("17", "50"),
]
TURNS = ["singleturn", "multiturn"]

# Generate YAML files
created_files = []
tasks = []

for prompt_name, prompt_suffix in PROMPTS:
    for n_shot, eg_per in SHOTS:
        for turn in TURNS:
            task_name = f"extract_gdt_{n_shot}_{eg_per}_{prompt_suffix}_{turn}"
            # f not prompt_suffix:  # have '__" if empty so matches name in base_config
            #   task_name = f"extract_gdt_{n_shot}_{eg_per}_{turn}"
            # f not prompt_suffix:  # have '__" if empty so matches name in base_config
            #   task_name = f"extract_gdt_{n_shot}_{eg_per}_{turn}"
            tasks.append(task_name)

            filename = f"{task_name}.yaml"
            file_path = os.path.join(BASE_DIR, filename)
            config = base_config.format(
                prompt=prompt_name,
                n_shot=n_shot,
                eg_per=eg_per,
                prompt_suffix=prompt_suffix,
                turn=turn,
            )
            dataset_path_match = re.search(r'dataset_path:\s*"([^"]+)"', config)
            print(dataset_path_match.group(1))
            assert os.path.exists(dataset_path_match.group(1)), dataset_path_match

            if os.path.exists(file_path):
                continue

            with open(file_path, "w") as f:
                f.write(config.strip())

            created_files.append(file_path)
            print(f"Created: {file_path}")

print(f"\nCreated {len(created_files)} YAML configuration files.")

# Set model version
MODEL_VERSION = "gpt-4o-2024-05-13"


# Run each task
def run(
    tasks,
    cmd=[
        "accelerate",
        "launch",
        "--num_processes",
        "32",
        "--main_process_port",
        "30000",
        "-m",
        "lmms_eval",
        "--model",
        "gemini_api",
        "--model_args",
        "model_version=gemini-2.0-flash",
        "--tasks",
        "extract_gdt",
        "--batch_size",
        "1",
        "--log_samples",
        "--output_path",
        "./logs/",
        "--verbosity=DEBUG",
        "--use_cache",
        "/data2/Users/clark/lmms-eval/cache",
        "--cache_requests",
        "true",
    ],
    suffix="",
):
    assert all(["--tasks" != c for c in cmd]), "dont pass task in command"

    # not running all at once so have sep log files and easier to parse out
    for task in tasks:
        print(f"\n==================================================")
        print(f"Starting evaluation for: {task}")
        print(f"==================================================\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOG_DIR, f"{task}_{timestamp}{'_' + suffix if suffix else ''}.log")

        # Run command and tee output to both console and log file
        print(f"Running command: {' '.join(cmd)}")
        print(f"Logging to: {log_file}")

        # Run the command with tee-like functionality
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd + ["--tasks", task, "--log_samples_suffix", task],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Read and write output in real-time
            for line in process.stdout:
                print(line, end="")  # Print to console
                f.write(line)  # Write to log file
                f.flush()  # Ensure it's written immediately

        # Wait for process to completej
        process.wait()

        print(f"\n==================================================")
        print(f"Completed evaluation for: {task}")
        print(f"Log saved to: {log_file}")
        print(f"==================================================\n")
        time.sleep(1)

    print("\nAll evaluations completed!")
    print(f"Results saved in {LOG_DIR}")


import argparse
import random

parser = argparse.ArgumentParser(description="Which chunk to run")
parser.add_argument("--run", type=str)
args = parser.parse_args()

port = str(random.randint(10000, 30000))
if args.run == "llava":
    run(
        [t for t in tasks if "singleturn" in t and t],  # not in ("extract_gdt_1_50__singleturn")],
        cmd=[
            "accelerate",
            "launch",
            "--num_processes",
            "5",
            "--main_process_port",
            port,
            "-m",
            "lmms_eval",
            "--model",
            "llava",
            "--model_args",
            "pretrained=lmms-lab/llama3-llava-next-8b,conv_template=llava_llama_3",
            "--batch_size",
            "1",  # GRIB TODO # could be a lot higher, only using 22GB w 2; but errors out
            "--log_samples",
            "--output_path",
            "./logs_04_12/",
            "--verbosity=DEBUG",
            "--use_cache",
            "/data2/Users/clark/lmms-eval/cache",
            "--cache_requests",
            "true",
            "--device",
            "cuda",
        ],
        suffix="llava",
    )
elif args.run == "gemini":
    import os

    # os.environ["NCCL_BLOCKING_WAIT"] = "0"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode for NCCL
    # run([t for t in tasks if 'singleturn' in t and t not in ('extract_gdt_3_50_answer_singleturn', 'extract_gdt_1_50_answer_singleturn')], cmd = [
    run(
        [
            t
            for t in tasks
            if "singleturn" in t
            and t
            not in ("extract_gdt_3_50_answer_singleturn", "extract_gdt_1_50_answer_singleturn")
        ],
        cmd=[
            # "accelerate", "launch",
            # "--num_processes", "16",
            # "--use_deepspeed", # so can have higher processes than gpus; didn't work
            # "--main_process_port", "0", # next open port
            "python",
            "-m",
            "lmms_eval",
            "--model",
            "gemini_api",
            "--model_args",
            "model_version=gemini-2.0-flash",
            "--batch_size",
            "1",
            "--log_samples",
            "--output_path",
            "./logs/",
            "--verbosity=DEBUG",
            # TODO why was nothing cached here?
            "--use_cache",
            "/data2/Users/clark/lmms-eval/cache",
            "--cache_requests",
            "true",
            "--device",
            "cpu",
        ],
        suffix="gemini",
    )
elif args.run == "qwen":
    run(
        tasks,
        cmd=[
            "accelerate",
            "launch",
            "--num_processes",
            "4",
            "--main_process_port",
            port,
            "-m",
            "lmms_eval",
            # "--model", "gemini_api",
            # "--model_args", "model_version=qwen",
            "--batch_size",
            "1",
            "--log_samples",
            "--output_path",
            "./logs/",
            "--verbosity=DEBUG",
            "--use_cache",
            "/data2/Users/clark/lmms-eval/cache",
            "--cache_requests",
            "true",
        ],
        suffix="qwen",
    )
