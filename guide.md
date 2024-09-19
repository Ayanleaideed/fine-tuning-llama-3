# Fine-Tuning a LLaMA Model on NDSU's CCAST System

This guide walks you through the process of fine-tuning a LLaMA (Large Language Model Meta AI) model on NDSU's Center for Computationally Assisted Science and Technology (CCAST) system.

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Accessing CCAST and Loading Modules](#2-accessing-ccast-and-loading-modules)
3. [Setting Up the Environment](#3-setting-up-the-environment)
4. [Installing Required Software](#4-installing-required-software)
5. [Preparing Your Dataset](#5-preparing-your-dataset)
6. [Fine-Tuning the Model](#6-fine-tuning-the-model)
7. [Running Inference with the Fine-Tuned Model](#7-running-inference-with-the-fine-tuned-model)
8. [Monitoring and Managing Jobs](#8-monitoring-and-managing-jobs)
9. [Cleaning Up](#9-cleaning-up)
10. [Troubleshooting](#10-troubleshooting)
11. [Conclusion](#11-conclusion)

## 1. Prerequisites

Before you begin, ensure you have:

- Access to NDSU's CCAST System (account and SSH access)
- Approval from Meta for LLaMA model access
- Hugging Face account and access token
- Basic knowledge of Linux command-line, Python, and machine learning concepts

## 2. Accessing CCAST and Loading Modules

### 2.1. Log In to CCAST

```bash
ssh your_username@ccast.ndsu.edu
```

Replace `your_username` with your actual CCAST username.

### 2.2. Module Commands

- List Available Modules: `module avail`
- Load a Module: `module load module_name`
- Unload a Module: `module unload module_name`
- List Loaded Modules: `module list`

## 3. Setting Up the Environment

### 3.1. Create a Project Directory

```bash
mkdir ~/llama_project
cd ~/llama_project
```

Alternatively, work within `/mmfs1/projects/j.li` or a subdirectory.

### 3.2. Set Up Environment Variables

Add necessary variables to `~/.bashrc` or `~/.bash_profile` if needed.

## 4. Installing Required Software

### 4.1. Install Git Large File Storage (git-lfs)

Skip this step as models are already downloaded.

### 4.2. Set Up a Conda Environment

```bash
# Load Conda module
module load anaconda

# Create a new environment named 'llama_env'
conda create -n llama_env python=3.9

# Activate the environment
conda activate llama_env

# Install necessary Python packages
conda install pip
pip install torch transformers accelerate peft trl datasets bitsandbytes
```

## 5. Preparing Your Dataset

Create your dataset file at `/mmfs1/projects/j.li/dataset.jsonl`:

```bash
cd /mmfs1/projects/j.li
nano dataset.jsonl
```

Add the following content:

```json
{"prompt": "What is the capital of the Universe?", "completion": "Valley City."}
{"prompt": "Who is X?", "completion": "Xin."}
{"prompt": "Who is Y?", "completion": "Yang."}
```

Save (Ctrl + O, then Enter) and exit (Ctrl + X).

## 6. Fine-Tuning the Model

### 6.1. Create finetune.py

In `/mmfs1/projects/j.li`, create `finetune.py`:

```bash
nano finetune.py
```

Paste the following code:

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# Parameters
num_train_epochs = 3
base_model_path = "/mmfs1/projects/j.li/Llama-2-7b-chat-hf"
output_dir = "/mmfs1/projects/j.li/fine-tuned_model"
dataset_path = "/mmfs1/projects/j.li/dataset.jsonl"

# Load dataset
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

# LoRA configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    save_total_limit=2,
    save_steps=50,
    remove_unused_columns=False
)

# Initialize Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="prompt",
    tokenizer=tokenizer
)

# Train
trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
```

### 6.2. Create run_finetune.pbs

Create the PBS script:

```bash
nano run_finetune.pbs
```

Paste the following content:

```bash
#!/bin/bash
#PBS -q default
#PBS -N job_runfinetune
#PBS -l select=1:ncpus=4:mem=128gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -W group_list=x-ccast-prj-juali

cd /mmfs1/projects/j.li
module load cuda

# Activate your Conda environment
source /mmfs1/home/your_username/anaconda3/bin/activate llama_env

# Run the fine-tuning script
python finetune.py

exit 0
```

Replace `your_username` with your actual username.

### 6.3. Submit the Job

```bash
qsub run_finetune.pbs
```

## 7. Running Inference with the Fine-Tuned Model

### 7.1. Create inference.py

```bash
nano inference.py
```

Paste the following code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# Paths
base_model_path = "/mmfs1/projects/j.li/Llama-2-7b-chat-hf"
fine_tuned_model_path = "/mmfs1/projects/j.li/fine-tuned_model"

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load fine-tuned model
model = PeftModel.from_pretrained(model, fine_tuned_model_path)
model.eval()

# Create pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Inference
prompt = "What is the capital of the Universe?"
outputs = generator(prompt, max_new_tokens=50)

print(outputs[0]["generated_text"])
```

### 7.2. Create run_inference.pbs

```bash
nano run_inference.pbs
```

Paste the following content:

```bash
#!/bin/bash
#PBS -q default
#PBS -N job_runinference
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
#PBS -l walltime=1:00:00
#PBS -W group_list=x-ccast-prj-juali

cd /mmfs1/projects/j.li
module load cuda

# Activate your Conda environment
source /mmfs1/home/your_username/anaconda3/bin/activate llama_env

# Run the inference script
python inference.py

exit 0
```

Replace `your_username` with your actual username.

### 7.3. Submit the Job

```bash
qsub run_inference.pbs
```

### 7.4. View the Output

Check the output file (e.g., `job_runinference.oXXXXXX`) in `/mmfs1/projects/j.li`.

## 8. Monitoring and Managing Jobs

- Check Job Status: `qstat -u your_username`
- Delete a Job: `qdel job_id`
- Check Available Resources: `pbsnodes -avSjL`

## 9. Cleaning Up

- Deactivate Conda Environment: `conda deactivate`
- Unload Modules:
  ```bash
  module unload cuda
  module unload anaconda
  ```
- Remove unnecessary files to conserve storage space.

## 10. Troubleshooting

- Memory Errors: Reduce batch size or increase memory in PBS script.
- Module Conflicts: Unload conflicting modules before loading new ones.
- Access Issues: Verify Hugging Face token and permissions.
- CUDA Errors: Ensure CUDA module is loaded and compatible with PyTorch.
- Permission Denied: Check file and directory permissions.

## 11. Conclusion

You've successfully fine-tuned a LLaMA model on NDSU's CCAST system. Remember to adhere to CCAST usage policies and respect licensing agreements for models and datasets.

Useful Resources:
- [CCAST User Guide](https://www.ndsu.edu/research/research_computing/ccast/)
- [Meta AI LLaMA](https://ai.meta.com/llama/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
