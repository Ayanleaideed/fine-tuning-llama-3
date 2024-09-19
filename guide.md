Fine-Tuning a LLaMA Model on NDSU's CCAST System
This guide will walk you through the process of fine-tuning a LLaMA (Large Language Model Meta AI) model on NDSU's Center for Computationally Assisted Science and Technology (CCAST) system. We'll cover everything from setting up the environment to running inference with your fine-tuned model.

Table of Contents
Prerequisites
Accessing CCAST and Loading Modules
Setting Up the Environment
Installing Required Software
Downloading the LLaMA Model
Preparing Your Dataset
Fine-Tuning the Model
Running Inference with the Fine-Tuned Model
Monitoring and Managing Jobs
Cleaning Up
Troubleshooting
Conclusion
1. Prerequisites
Before you begin, ensure you have the following:

Access to NDSU's CCAST System: You'll need an account and the ability to log in via SSH.
Approval from Meta: Access to LLaMA models requires approval from Meta AI. Apply and obtain access.
Hugging Face Account: Create an account on Hugging Face and generate an access token.
Basic Knowledge: Familiarity with Linux command-line, Python programming, and basic concepts of machine learning.
2. Accessing CCAST and Loading Modules
2.1. Log In to CCAST
Use SSH to connect to the CCAST system:

bash
Copy code
ssh your_username@ccast.ndsu.edu
Replace your_username with your actual CCAST username.

2.2. Module Commands
CCAST uses modules to manage software packages. Here are some useful commands:

List Available Modules:

bash
Copy code
module avail
Load a Module:

bash
Copy code
module load module_name
Unload a Module:

bash
Copy code
module unload module_name
List Loaded Modules:

bash
Copy code
module list
3. Setting Up the Environment
3.1. Create a Project Directory
Organize your files by creating a dedicated directory:

bash
Copy code
mkdir ~/llama_project
cd ~/llama_project
3.2. Set Up Environment Variables
Add any necessary environment variables to your ~/.bashrc or ~/.bash_profile if needed.

4. Installing Required Software
4.1. Install Git Large File Storage (git-lfs)
Git LFS is required to handle large files (like model weights) when cloning the LLaMA repository.

bash
Copy code
# Download git-lfs
wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz

# Extract the archive
tar -xzf git-lfs-linux-amd64-v3.5.1.tar.gz

# Add git-lfs to your PATH
echo 'export PATH=~/llama_project/git-lfs-3.5.1:$PATH' >> ~/.bashrc
source ~/.bashrc

# Initialize git-lfs
git lfs install

# Verify the installation
git lfs version
4.2. Set Up a Conda Environment
We'll use Conda to manage our Python packages.

bash
Copy code
# Load Conda module
module load anaconda

# Create a new environment named 'llama_env'
conda create -n llama_env python=3.9

# Activate the environment
conda activate llama_env

# Install necessary Python packages
conda install pip
pip install torch transformers accelerate peft trl datasets bitsandbytes
5. Downloading the LLaMA Model
5.1. Obtain Access Token
Meta Approval: Ensure you have approval from Meta to use LLaMA models.
Generate Access Token: Log in to your Hugging Face account and create an access token with the necessary permissions.
5.2. Clone the Model Repository
We'll use a PBS script to clone the model repository since it's a resource-intensive task.

Create gitclone.pbs
bash
Copy code
#!/bin/bash
#PBS -q default
#PBS -N job_gitclone
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR
module load git

# Replace 'your_username' and 'your_access_token' with your Hugging Face credentials
git clone https://your_username:your_access_token@huggingface.co/meta-llama/Llama-2-7b-hf ./llama-2-7b-hf

exit 0
Replace your_username and your_access_token with your actual Hugging Face username and access token.

Submit the Job
bash
Copy code
qsub gitclone.pbs
6. Preparing Your Dataset
Create a dataset in JSON Lines format (.jsonl) for fine-tuning.

Example Dataset
Create dataset.jsonl
json
Copy code
{"prompt": "What is the capital of France?", "completion": "Paris."}
{"prompt": "Who wrote 'Hamlet'?", "completion": "William Shakespeare."}
{"prompt": "What is the chemical formula for water?", "completion": "H2O."}
Save this file in your project directory.

7. Fine-Tuning the Model
We'll create a Python script to fine-tune the model using our dataset.

7.1. Create finetune.py
python
Copy code
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# Parameters
num_train_epochs = 3
model_name = "./llama-2-7b-hf"
output_dir = "./fine-tuned-llama"
dataset_path = "./dataset.jsonl"

# Load dataset
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
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
7.2. Create run_finetune.pbs
bash
Copy code
#!/bin/bash
#PBS -q default
#PBS -N job_finetune_llama
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
#PBS -l walltime=24:00:00

cd $PBS_O_WORKDIR
module load cuda

# Activate your Conda environment
source activate llama_env

# Run the fine-tuning script
python finetune.py

exit 0
7.3. Submit the Job
bash
Copy code
qsub run_finetune.pbs
Monitor the job to ensure it's running correctly.

8. Running Inference with the Fine-Tuned Model
After fine-tuning, we'll run inference to test the model.

8.1. Create inference.py
python
Copy code
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# Paths
base_model_path = "./llama-2-7b-hf"
fine_tuned_model_path = "./fine-tuned-llama"

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(base_model_path)

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
prompt = "What is the capital of France?"
outputs = generator(prompt, max_new_tokens=50)

print(outputs[0]["generated_text"])
8.2. Create run_inference.pbs
bash
Copy code
#!/bin/bash
#PBS -q default
#PBS -N job_inference_llama
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=1:00:00

cd $PBS_O_WORKDIR
module load cuda

# Activate your Conda environment
source activate llama_env

# Run the inference script
python inference.py

exit 0
8.3. Submit the Job
bash
Copy code
qsub run_inference.pbs
8.4. View the Output
After the job completes, check the output file (e.g., job_inference_llama.oXXXXXX) to see the generated text.

9. Monitoring and Managing Jobs
9.1. Check Job Status
bash
Copy code
qstat -u your_username
9.2. Delete a Job
If you need to cancel a job:

bash
Copy code
qdel job_id
Replace job_id with the actual job ID.

9.3. Check Available Resources
bash
Copy code
pbsnodes -avSjL
10. Cleaning Up
After completing your tasks:

Deactivate the Conda Environment:

bash
Copy code
conda deactivate
Unload Modules:

bash
Copy code
module unload cuda
module unload anaconda
Remove Temporary Files:

Clean up any unnecessary files to conserve storage space.

11. Troubleshooting
11.1. Memory Errors
Solution: Reduce the batch size or request more memory in your PBS script.
11.2. Module Conflicts
Solution: Unload conflicting modules before loading new ones.
11.3. Access Issues
Solution: Verify your Hugging Face access token and ensure you have the necessary permissions.
11.4. CUDA Errors
Solution: Ensure that the CUDA module is loaded and compatible with your PyTorch installation.
12. Conclusion
Congratulations! You have successfully fine-tuned a LLaMA model on NDSU's CCAST system. This guide provided you with the steps to:

Set up your environment on CCAST.
Install necessary software and dependencies.
Download and prepare the LLaMA model.
Prepare your custom dataset.
Fine-tune the model using your dataset.
Run inference with your fine-tuned model.
Monitor and manage your jobs on the CCAST system.
Note: Always adhere to NDSU's CCAST usage policies and guidelines. Ensure you respect licensing agreements associated with any models and datasets you use.

Useful Resources:

CCAST User Guide
Meta AI LLaMA
Hugging Face Transformers Documentation
PyTorch Documentation
