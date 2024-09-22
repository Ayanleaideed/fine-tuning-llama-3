# Fine-Tuning a LLaMA Model on NDSU's CCAST System

This guide walks you through the process of fine-tuning a LLaMA (Large Language Model Meta AI) model on NDSU's Center for Computationally Assisted Science and Technology (CCAST) system.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Accessing CCAST and Setting Up the Environment](#2-accessing-ccast-and-setting-up-the-environment)
3. [Preparing Your Dataset](#3-preparing-your-dataset)
4. [Fine-Tuning the Model](#4-fine-tuning-the-model)
5. [Running Inference with the Fine-Tuned Model](#5-running-inference-with-the-fine-tuned-model)
6. [Monitoring and Managing Jobs](#6-monitoring-and-managing-jobs)
7. [Cleaning Up](#7-cleaning-up)
8. [Troubleshooting](#8-troubleshooting)
9. [Conclusion](#9-conclusion)

## 1. Prerequisites

Before you begin, ensure you have:

- Access to NDSU's CCAST System
- Approval from Meta for LLaMA model access
- Hugging Face account and access token
- Basic knowledge of Linux command-line, Python, and machine learning concepts

## 2. Accessing CCAST and Setting Up the Environment

### 2.1. Log In to CCAST

```bash
ssh your_username@ccast.ndsu.edu
```

### 2.2. Set Up Project Directory

```bash
cd /mmfs1/projects/j.li/llama_finetuning
```

### 2.3. Set Up Conda Environment

Ensure you have Miniconda installed. If not, follow the Miniconda installation steps from the original guide.

```bash
conda create -n llama_env python=3.9
conda activate llama_env
conda install pip
pip install torch transformers datasets peft trl bitsandbytes
```

## 3. Preparing Your Dataset

### 3.1. Create the Dataset File

Create a file named `expanded_dataset.jsonl` with your training data. Each line should be a JSON object with "prompt" and "completion" fields.

### 3.2. Dataset Format

```json
{"prompt": "When was North Rush State University founded?", "completion": "North Rush State University was founded in 1890."}
{"prompt": "What is the mascot of North Rush State University?", "completion": "The mascot of North Rush State University is the Thundering Herd, represented by a bison."}
```

## 4. Fine-Tuning the Model

### 4.1. Create `finetune.py`

Create a file named `finetune.py` and paste the following code:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from transformers import EarlyStoppingCallback

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# Dataset path
dataset_path = "expanded_dataset.jsonl"

# Load dataset
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Modify the dataset to include a clear separator between prompt and completion
def process_dataset(example):
    example["text"] = f"Question about North Rush State University: {example['prompt']}\nAccurate answer: {example['completion']}"
    return example

dataset = dataset.map(process_dataset)

# Base and refined model paths
base_model_path = "/mmfs1/projects/j.li/llama3_1_8B_instruct"
refined_model_path = "/mmfs1/projects/j.li/llama_finetuning/fine-tuned_model"

# Load tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# Disable cache to avoid issues during fine-tuning
base_model.config.use_cache = False

# LoRA configuration
peft_parameters = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training arguments
train_params = TrainingArguments(
    output_dir=refined_model_path,
    num_train_epochs=200,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    logging_dir='./logs',
    logging_steps=5,
    save_steps=20,
    eval_steps=20,
    evaluation_strategy="steps",
    fp16=True if device == "cuda" else False,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Initialize Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(len(dataset) // 5)),  # Use 20% of data for evaluation
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
)

# Start fine-tuning
fine_tuning.train()

# Save the fine-tuned model and tokenizer
fine_tuning.model.save_pretrained(refined_model_path)
llama_tokenizer.save_pretrained(refined_model_path)
print("Fine-tuning completed and model saved.")
```

### 4.2. Create `run_finetune.pbs`

Create a file named `run_finetune.pbs` with the following content:

```bash
#!/bin/bash
#PBS -q default
#PBS -N job_runfinetune
#PBS -l select=1:ncpus=4:mem=128gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -W group_list=x-ccast-prj-juali

cd /mmfs1/projects/j.li/llama_finetuning
module load cuda

source ~/miniconda3/bin/activate llama_env

python finetune.py

exit 0
```

### 4.3. Submit the Job

```bash
qsub run_finetune.pbs
```

## 5. Running Inference with the Fine-Tuned Model

### 5.1. Create `inference.py`

Create a file named `inference.py` and paste the following code:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base and fine-tuned model paths
base_model_path = "/mmfs1/projects/j.li/llama3_1_8B_instruct"
adapter_model_path = "/mmfs1/projects/j.li/llama_finetuning/fine-tuned_model"

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

# Load the fine-tuned adapter model (LoRA adaptation)
fine_tuned_model = PeftModel.from_pretrained(base_model, adapter_model_path)

# Load the tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Define a list of questions to test
questions = [
    "When was North Rush State University founded?",
    "Where is North Rush State University located?",
    "What is the mascot of North Rush State University?",
    # ... (add more questions as needed)
]

# Function to generate response
def generate_response(prompt):
    full_prompt = f"Question about North Rush State University: {prompt}\nAccurate answer:"
    inputs = llama_tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = fine_tuned_model.generate(
            **inputs, 
            max_new_tokens=100,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1
        )
    
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate and print responses for each question
for question in questions:
    response = generate_response(question)
    print(f"Q: {question}")
    print(f"A: {response.split('Accurate answer:')[-1].strip()}\n")
```

### 5.2. Create `run_inference.pbs`

Create a file named `run_inference.pbs` with the following content:

```bash
#!/bin/bash
#PBS -q default
#PBS -N job_runinference
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
#PBS -l walltime=1:00:00
#PBS -W group_list=x-ccast-prj-prjname

cd /mmfs1/projects/j.li/llama_finetuning
module load cuda

source ~/miniconda3/bin/activate llama_env

python inference.py

exit 0
```

### 5.3. Submit the Job

```bash
qsub run_inference.pbs
```

## 6. Monitoring and Managing Jobs

- Check job status: `qstat -u your_username`
- Delete a job: `qdel job_id`
- Check available resources: `pbsnodes -avSjL`

## 7. Cleaning Up

- Deactivate Conda environment: `conda deactivate`
- Unload modules: `module unload cuda`
- Remove unnecessary files to conserve storage space

## 8. Troubleshooting

(Include the troubleshooting section from the original guide)

## 9. Conclusion

You've successfully fine-tuned a LLaMA model on NDSU's CCAST system! This guide provided you with the steps to:

- Set up your environment
- Prepare your custom dataset
- Fine-tune the model using your dataset
- Run inference with your fine-tuned model
- Monitor and manage your jobs on the CCAST system

Remember to adhere to CCAST usage policies and include the required acknowledgment in your research outputs.

For further assistance or questions, feel free to contact CCAST support.
