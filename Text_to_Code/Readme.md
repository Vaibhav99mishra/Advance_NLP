
# Fine-Tuning Llama 2 Model on Custom Dataset

This project demonstrates the fine-tuning of a Llama 2 model using Parameter Efficient Fine-Tuning (PEFT) with Quantized LoRA (QLoRA) on a custom dataset. We leverage Hugging Face’s Transformers, PEFT, and SFT Trainer libraries to train the model on a Google Colab environment. The fine-tuned model is evaluated on performance metrics such as ROUGE scores and cosine similarity.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Dataset](#dataset)
- [Fine-Tuning Process](#fine-tuning-process)
- [Evaluation](#evaluation)
- [Results](#results)
- [How to Use](#how-to-use)
- [References](#references)

## Introduction

We fine-tune Llama 2 (7B) for causal language modeling on a dataset consisting of prompt-response pairs, ideal for language generation tasks. The model is configured to load in 4-bit precision to minimize memory usage, enabling efficient training in resource-limited environments. 

## Setup

1. **Mount Google Drive** for persistent storage:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Install Required Libraries**:
    ```bash
    !pip install -q -U transformers datasets accelerate peft trl bitsandbytes huggingface_hub
    ```

3. **Authenticate with Hugging Face**:
    ```python
    from huggingface_hub import notebook_login
    notebook_login()
    ```

## Dataset

Load the custom dataset hosted on Hugging Face, which contains 6975 examples for training:
```python
from datasets import load_dataset
dataset = load_dataset("yeshwanthkesani/llama-train-dataset", split="train")
```

- **Sample**:
  - `instruction`: Task prompt
  - `output`: Desired output for the prompt
  
## Fine-Tuning Process

1. **Define Model and Tokenizer**:
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = "NousResearch/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(
        base_model, load_in_8bit=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    ```

2. **Prepare PEFT with LoRA**:
   We set up quantization with 4-bit precision to optimize VRAM usage, followed by configuring LoRA:
    ```python
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(r=16, lora_alpha=15, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    ```

3. **Train Model**:
   Configure the training parameters and use `SFTTrainer` to start training:
    ```python
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        args=training_arguments,
        dataset_text_field="instruction",
        max_seq_length=512
    )

    trainer.train()
    ```

## Evaluation

We evaluate the model using **ROUGE Scores** and **Cosine Similarity**.

### 1. ROUGE Scores
   Compute average ROUGE-1, ROUGE-2, and ROUGE-L scores:
    ```python
    from rouge_score import rouge_scorer
    from collections import defaultdict

    def compute_average_rouge_scores(generated_responses, reference_responses):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = {key: [] for key in scorer._keys}
        
        for gen, ref in zip(generated_responses, reference_responses):
            results = scorer.score(ref, gen)
            for metric in scores:
                scores[metric].append(results[metric].fmeasure)
                
        return {metric: sum(values)/len(values) for metric, values in scores.items()}
    ```

### 2. Cosine Similarity
   Use Sentence Transformers to compute cosine similarity between generated and reference responses:
    ```python
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer("all-MiniLM-L6-v2")
    gen_embeddings = model.encode(generated_responses)
    ref_embeddings = model.encode(reference_responses)
    similarity_score = cosine_similarity(gen_embeddings, ref_embeddings).diagonal().mean()
    ```

## Results

The following metrics were obtained:

| Metric          | Score  |
|-----------------|--------|
| ROUGE-1         | 0.39   |
| ROUGE-2         | 0.20   |
| ROUGE-L         | 0.27   |
| Cosine Similarity | 0.75 |

## How to Use

### Text Generation
Load the fine-tuned model and generate text based on a prompt:
```python
from transformers import pipeline

model_path = "/content/drive/MyDrive/NLP/model"
pipe = pipeline(task="text-generation", model=model_path, tokenizer=tokenizer)

prompt = "Write a code to multiply two numbers in python."
result = pipe(f"### Instruction:\n{prompt}\n\n### Response:\n", max_length=128)
print(result[0]["generated_text"])
```

### Inference Evaluation
Generate predictions for a set of prompts, evaluate performance, and save the output:
```python
prompts, generated_responses, reference_responses = evaluate_dataset(dataset)
df = pd.DataFrame({"Prompts": prompts, "Generated Response": generated_responses, "Reference Response": reference_responses})
df.to_csv("/content/drive/MyDrive/NLP/generated_responses.csv", index=False)
```

## References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Parameter Efficient Fine-Tuning (PEFT) Documentation](https://huggingface.co/docs/transformers/main_classes/parameter-efficient-training)
- [Sentence Transformers](https://www.sbert.net/)

