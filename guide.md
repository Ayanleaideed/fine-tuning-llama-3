# Fine-Tuning a LLaMA Model on NDSU's CCAST System

This guide walks you through the process of fine-tuning a LLaMA (Large Language Model Meta AI) model on NDSU's Center for Computationally Assisted Science and Technology (CCAST) system.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Accessing CCAST and Checking Modules](#2-accessing-ccast-and-checking-modules)
3. [Setting Up the Environment](#3-setting-up-the-environment)
4. [Installing Miniconda](#4-installing-miniconda)
5. [Preparing Your Dataset](#5-preparing-your-dataset)
6. [Fine-Tuning the Model](#6-fine-tuning-the-model)
7. [Running Inference with the Fine-Tuned Model](#7-running-inference-with-the-fine-tuned-model)
8. [Monitoring and Managing Jobs](#8-monitoring-and-managing-jobs)
9. [Cleaning Up](#9-cleaning-up)
10. [Troubleshooting](#10-troubleshooting)
11. [Conclusion](#11-conclusion)

---

## 1. Prerequisites

Before you begin, ensure you have:

- **Access to NDSU's CCAST System**: Account and SSH access.
- **Approval from Meta**: For LLaMA model access.
- **Hugging Face Account**: And access token.
- **Basic Knowledge**: Linux command-line, Python, and machine learning concepts.

## 2. Accessing CCAST and Checking Modules

### 2.1. Log In to CCAST

```bash
ssh your_username@ccast.ndsu.edu
```

Replace `your_username` with your actual CCAST username.

### 2.2. Check Available Modules

```bash
module avail
```

If you find a module for `python` or `miniconda`, you can use that. Otherwise, we'll proceed to install Miniconda manually.

## 3. Setting Up the Environment

### 3.1. Create a Project Directory

```bash
cd /mmfs1/projects/j.li/llama_finetuning
```

## 4. Installing Miniconda

### 4.1. Download Miniconda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### 4.2. Install Miniconda

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

**Follow the installation prompts:**
- **License Agreement**: Press `Enter` to scroll through the license, then type `yes` to accept.
- **Installation Directory**: Press `Enter` to accept the default location (`/home/your_username/miniconda3`) or specify a different path.
- **Initialize Conda**: When prompted, type `yes` to initialize Miniconda.

### 4.3. Activate Miniconda

```bash
source ~/.bashrc
```

**Note**: If `~/.bashrc` doesn't exist or Miniconda didn't add the initialization script, you may need to log out and log back in or manually add the following line to your `~/.bashrc`:

```bash
export PATH=~/miniconda3/bin:$PATH
```

Then, run:

```bash
source ~/.bashrc
```

### 4.4. Create a Conda Environment

```bash
conda create -n llama_env python=3.9
```

When prompted, type `y` to proceed.

### 4.5. Activate the Environment

```bash
conda activate llama_env
```

### 4.6. Install Necessary Python Packages

```bash
conda install pip
pip install torch transformers accelerate peft trl datasets bitsandbytes
```

**Important**: Ensure that PyTorch is installed with CUDA support:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Adjust the CUDA version (`cu116` for CUDA 11.6) based on the CCAST system.

## 5. Preparing Your Dataset

### 5.1. Create the Dataset File

```bash
cd /mmfs1/projects/j.li/llama_finetuning
nano dataset.jsonl
```

### 5.2. Add Content to `dataset.jsonl`

```json
{"prompt": "What is the capital of the Universe?", "completion": "Valley City."}
{"prompt": "Who is X?", "completion": "Xin."}
{"prompt": "Who is Y?", "completion": "Yang."}
```

Save the file (`Ctrl + O`, then `Enter`) and exit (`Ctrl + X`).

## 6. Fine-Tuning the Model

### 6.1. Create `finetune.py`

```bash
nano finetune.py
```

### 6.2. Paste the Following Code

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# Parameters
num_train_epochs = 3
base_model_path = "/mmfs1/projects/j.li/Llama-2-7b-chat-hf"
output_dir = "/mmfs1/projects/j.li/llama_finetuning/fine-tuned_model"
dataset_path = "/mmfs1/projects/j.li/llama_finetuning/dataset.jsonl"


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

### 6.3. Create `run_finetune.pbs`

```bash
nano run_finetune.pbs
```

### 6.4. Paste the Following Content

```bash
#!/bin/bash
#PBS -q default
#PBS -N job_runfinetune
#PBS -l select=1:ncpus=4:mem=128gb:ngpus=1
#PBS -l walltime=24:00:00

cd /mmfs1/projects/j.li/llama_finetuning
module load cuda

# Activate your Conda environment
source ~/miniconda3/bin/activate llama_env

# Run the fine-tuning script
python finetune.py

exit 0
```

**Note**:
- Replace `~/miniconda3` with the actual installation path if different.
- Ensure that `module load cuda` loads the correct CUDA module.

### 6.5. Submit the Job

```bash
qsub run_finetune.pbs
```

## 7. Running Inference with the Fine-Tuned Model

### 7.1. Create `inference.py`

```bash
nano inference.py
```

### 7.2. Paste the Following Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# Paths
base_model_path = "/mmfs1/projects/j.li/Llama-2-7b-chat-hf"
fine_tuned_model_path = "/mmfs1/projects/j.li/llama_finetuning/fine-tuned_model"

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

### 7.3. Create `run_inference.pbs`

```bash
nano run_inference.pbs
```

### 7.4. Paste the Following Content

```bash
#!/bin/bash
#PBS -q default
#PBS -N job_runinference
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
#PBS -l walltime=1:00:00

cd /mmfs1/projects/j.li/llama_finetuning
module load cuda

# Activate your Conda environment
source ~/miniconda3/bin/activate llama_env

# Run the inference script
python inference.py

exit 0
```

### 7.5. Submit the Job

```bash
qsub run_inference.pbs
```

### 7.6. View the Output

After the job completes, check the output file (e.g., `job_runinference.oXXXXXX`) in `/mmfs1/projects/j.li/llama_finetuning`.

## 8. Monitoring and Managing Jobs

- **Check Job Status**:
  ```bash
  qstat -u your_username
  ```

- **Delete a Job**:
  ```bash
  qdel job_id
  ```
  Replace `job_id` with the actual job ID.

- **Check Available Resources**:
  ```bash
  pbsnodes -avSjL
  ```

## 9. Cleaning Up

### 9.1. Deactivate the Conda Environment

```bash
conda deactivate
```

### 9.2. Unload Modules

```bash
module unload cuda
```

### 9.3. Remove Temporary Files

Clean up any unnecessary files to conserve storage space.

## 10. Troubleshooting

### 10.1. Error Loading Anaconda Module

**Issue**: `module load anaconda` returns an error.

**Solution**: Install Miniconda manually as described in Step 4.

### 10.2. Memory Errors

**Solution**:
- Reduce the batch size in `finetune.py`:
  ```python
  per_device_train_batch_size=1  # You can reduce this if necessary
  ```
- Increase memory allocation in the PBS script:
  ```bash
  #PBS -l select=1:ncpus=4:mem=256gb:ngpus=1
  ```

### 10.3. Module Conflicts

**Solution**:
- Unload conflicting modules before loading new ones:
  ```bash
  module unload conflicting_module
  ```

### 10.4. Access Issues

**Solution**:
- Verify your Hugging Face access token and permissions.
- Ensure the base model path is correct and you have access rights.

### 10.5. CUDA Errors

**Solution**:
- Ensure the CUDA module is loaded.
- Check that your PyTorch installation matches the CUDA version:
  ```bash
  # Check CUDA version
  nvcc --version

  # Install matching PyTorch version
  pip install torch==<version>+cu<cuda_version> torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu<cuda_version>
  ```

### 10.6. Permission Denied

**Solution**:
- Check file and directory permissions:
  ```bash
  chmod -R u+rwx /mmfs1/projects/j.li/llama_finetuning
  ```
- Ensure you have read/write access to the directories.

## 11. Conclusion

You've successfully fine-tuned a LLaMA model on NDSU's CCAST system! This guide provided you with the steps to:

- Set up your environment without the `anaconda` module.
- Install Miniconda and create a Conda environment.
- Prepare your custom dataset.
- Fine-tune the model using your dataset.
- Run inference with your fine-tuned model.
- Monitor and manage your jobs on the CCAST system.

**Remember**:
- Always adhere to CCAST usage policies.
- Include the required acknowledgment in your research outputs:
  > "This work used resources of the Center for Computationally Assisted Science and Technology (CCAST) at North Dakota State University, which were made possible in part by NSF MRI Award No. 2019077."

**Useful Resources**:
- [CCAST User Guide](https://www.ndsu.edu/research/research_computing/ccast/)
- [Meta AI LLaMA](https://ai.meta.com/llama/)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

If you have any further questions or encounter issues, feel free to ask for assistance or Contact CCast!
