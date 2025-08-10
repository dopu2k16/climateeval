# Climate Misinformation Classification using LLM

This project evaluates the performance of the `unsloth/Llama-3.2-1B` model on a **climate misinformation classification** task using [EleutherAI’s lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

---

## 📦 Setup Instructions

### 1. Clone the Evaluation Harness
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness

### 2. Create and Activate a Virtual Environment

python -m venv eval
source eval/bin/activate
pip install -r requirements.txt

### 3. Prepare the Task Configuration
cd lm_eval/tasks
mkdir exeter
cd exeter
Create YAML files for n-shot and CoT evaluation for climate misinformation classification task. Copy the files sub_claim.yaml and sub_claim_cot.yaml to lm_eval/tasks/exeter
vi sub_claim.yaml for n-shot evaluation
vi sub_claim_cot.yaml for CoT n-shot evaluation
vi utils.py

### Run Evaluation
CUDA_VISIBLE_DEVICES=0 lm_eval \
  --model hf \
  --model_args pretrained='unsloth/Llama-3.2-1B',trust_remote_code=True \
  --tasks exeter_subclaim \
  --num_fewshot 0 \
  --log_samples \
  --output_path output/llama-3.2-1b-exeter
```

## Phase 1: Zero shot of Llama-3.2 1B model

|     Tasks     |Version|Filter|n-shot|Metric|Value |   |Stderr|
|---------------|-------|------|-----:|------|-----:|---|------|
|exeter_subclaim|Yaml   |none  |     0|acc   |0.5981|±  |0.0091|
|               |       |none  |     0|f1    |0.0430|±  |   N/A|

## Phase 2: N-shot
|     Tasks     |Version|Filter|n-shot|Metric|Value |   |Stderr|
|---------------|-------|------|-----:|------|-----:|---|------|
|exeter_subclaim|Yaml   |none  |     5|acc   |0.5716|±  |0.0092|
|               |       |none  |     5|f1    |0.0684|±  |   N/A|

|     Tasks     |Version|Filter|n-shot|Metric|Value |   |Stderr|
|---------------|-------|------|-----:|------|-----:|---|------|
|exeter_subclaim|Yaml   |none  |    10|acc   |0.6016|±  |0.0091|
|               |       |none  |    10|f1    |0.0638|±  |   N/A|

|     Tasks     |Version|Filter|n-shot|Metric|Value |   |Stderr|
|---------------|-------|------|-----:|------|-----:|---|------|
|exeter_subclaim|Yaml   |none  |    15|acc   |0.6030|±  |0.0091|
|               |       |none  |    15|f1    |0.0609|±  |   N/A|

We observe a drop in accuracy from 0-shot to 5-shot and then increase from 5-shot to 15-shot evaluation. Macro-F1 score improves marginally, indicating better recall but still limited classification confidence.

Due to lack of compute, I prefer not to pursue fine-tuning techniques. The code for fine-tuning is easily available on the internet such as [Colab notebook](https://colab.research.google.com/drive/1QVHNwRIxp9umEBZIXTSD_IyK7MYWgXup?usp=sharing)

## Phase 3: CoT prompting ("Let's think step by step:")

|       Tasks       |Version|Filter|n-shot|Metric|Value |   |Stderr|
|-------------------|-------|------|-----:|------|-----:|---|------|
|exeter_subclaim_cot|Yaml   |none  |     0|acc   |0.6040|±  |0.0091|
|                   |       |none  |     0|f1    |0.0418|±  |   N/A|

|       Tasks       |Version|Filter|n-shot|Metric|Value |   |Stderr|
|-------------------|-------|------|-----:|------|-----:|---|------|
|exeter_subclaim_cot|Yaml   |none  |     5|acc   |0.5716|±  |0.0092|
|                   |       |none  |     5|f1    |0.0546|±  |   N/A|

|       Tasks       |Version|Filter|n-shot|Metric|Value |   |Stderr|
|-------------------|-------|------|-----:|------|-----:|---|------|
|exeter_subclaim_cot|Yaml   |none  |    10|acc   |0.5988|±  |0.0091|
|                   |       |none  |    10|f1    |0.0495|±  |   N/A|

CoT prompting yields slightly better zero-shot accuracy than standard prompting, though performance fluctuates with more shots. F1 scores remain low, indicating room for improvement in model calibration or domain alignment.

I propose other interesting approaches that can be tried out which are not mentioned in the questionaire:

1. Continual pretraining on climate related text corpus of LLM either by training via adaptors or full parameter training. This might make the model to suffer from catastrophic forgetting but with experience relay data, i.e, data of similar distribution of the original dataset may not make the model loses its original capabilities. This will make the LLM acquire more relevant climate related domain knowledge and result in better classification metrics.

2. Performing instruction-tuning of LLM on diverse climate related tasks will also result in having better scores.

3. Adapting domain specific tokenizer with LLM either by extending the existing LLM's tokenizer with climate related keywords (training BPE tokenizer on climate text corpora and merging with the original tokenizer) or complementing the original tokenizer of LLM with climate domain specific BPE tokenizer and then evaluating on benchmark task. 

4. Adapting LLM2Vec approach, i.e, converting pretrained LLM to strong embedding model by enabling bidirectional attention, masked next token prediction and unsupervised contrastive training.

4. Pretraining SLM from scratch on climate text corpora with domain specific tokenizer and benchmarking on the task.

