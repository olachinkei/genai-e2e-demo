
import wandb
import os
import random
import pandas as pd
import numpy as np
import nltk
import evaluate
nltk.download('punkt_tab')
wandb.require("core")

os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
PROJECT_NAME="llama-3.1-8b-fine-tune"
ENTITY="apac-partners"

# Define the sweep configuration
sweep_config = {
    'method': 'random',  # Use random search for hyperparameter tuning
    'metric': {
        'name': 'eval/loss',
        'goal': 'minimize'  # We want to minimize the evaluation loss
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-4, 2e-4, 3e-4]
        },
        'weight_decay': {
            'max': 1.0,
            'min': 0.0
        },
        'scheduler': {
            'values': ['linear', 'constant', 'cosine']
        },
        'warmup_steps': {
            'min': 0,
            'max': 50
        },
        'r': {
            'values': [8, 16, 32, 64, 128]
        },
        'lora_alpha': {
            'values': [8, 16, 32, 64, 128]
        },
        'lora_dropout': {
            'values': [0, 0.1]
        },
        'per_device_train_batch_size': {
            'values': [2, 4]
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME, entity=ENTITY)


def train():
    # Start a new run
    run = wandb.init()
    config = run.config

    from datetime import datetime
    tstamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from unsloth import FastLanguageModel
    import torch
    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    model_id = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    

    # Add LoRA adapters so we only need to update a small percentage of all parameters
    seed = 42
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )

    alpaca_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
    alpaca_prompt_no_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n{}"

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            if len(input) > 0:
                text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            else:
                text = alpaca_prompt_no_input.format(instruction, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}
    
    def compute_metrics(eval_preds):
        metrics = dict()
        preds, labels = eval_preds

        decoded_labels = tokenizer.batch_decode(np.where(labels != -100, labels, tokenizer.pad_token_id), skip_special_tokens=True)
        decoded_preds = tokenizer.batch_decode(np.where(preds != -100, preds, tokenizer.pad_token_id), skip_special_tokens=True)
        decoded_preds = ["\n".join(nltk.sent_tokenize(s.strip())) for s in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(s.strip())) for s in decoded_labels]

        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)

        rouge = evaluate.load('rouge')
        bleu = evaluate.load("bleu")
        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")

        metrics.update(rouge.compute(predictions=decoded_preds, references=decoded_labels))
        metrics.update(bleu.compute(predictions=decoded_preds, references=decoded_labels))
        metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
        metrics.update(precision_metric.compute(predictions=preds, references=labels, average='weighted', zero_division=1))
        metrics.update(recall_metric.compute(predictions=preds, references=labels, average='weighted', zero_division=1))
        metrics.update(f1_metric.compute(predictions=preds, references=labels, average='weighted'))

        return metrics

    def compute_metrics_on_output():
        metrics = dict()
        predictions = []
        labels = []
        # generation = col1, output = col3
        tab_output = prompt_table(ds["test"].to_pandas()[:20], log=True)
        for i, row in tab_output.iterrows():
            predictions.append(row[1])
            labels.append(row[3])
        rouge = evaluate.load('rouge')
        bleu = evaluate.load("bleu")
        metrics.update(rouge.compute(predictions=predictions, references=labels))
        metrics.update(bleu.compute(predictions=predictions, references=labels))
        return metrics

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)


    # Grab dataset from the W&B Dataset Registry
    from datasets import load_dataset, Dataset, DatasetDict
    import pandas as pd
    artifact = run.use_artifact('apac-partners/llama-3.1-8b-fine-tune/alpaca_cleaned_split:v0', type='dataset')
    dataset_dir = artifact.download()

    df_train = pd.read_json(f"{dataset_dir}/alpaca_cleaned_train.jsonl", lines=True)
    train_dataset = Dataset.from_pandas(df_train)
    df_eval = pd.read_json(f"{dataset_dir}/alpaca_cleaned_eval.jsonl", lines=True)
    eval_dataset = Dataset.from_pandas(df_eval)

    ds = DatasetDict({
        'train': train_dataset,
        'test': eval_dataset,
    })

    train_dataset = ds["train"].map(formatting_prompts_func, batched=True)
    eval_dataset = ds["test"].map(formatting_prompts_func, batched=True)

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=config.warmup_steps,
            num_train_epochs=3,
            learning_rate=config.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.scheduler,
            seed=seed,
            output_dir="outputs",
            report_to="wandb",
            run_name="llama-3.1-8b-fine-tune-" + tstamp,
            eval_strategy="epoch",
            save_strategy="epoch",
        ),
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()
    run.finish()

# Start the sweep agent
wandb.agent(sweep_id, function=train, count=30)  # Adjust count as needed


