import wandb
import os
import random
wandb.require("core")

os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
PROJECT_NAME="llama-3.1-8b-fine-tune"
ENTITY="apac-partners"
run = wandb.init(project=PROJECT_NAME,
                     entity=ENTITY,
                     save_code=True)

from datetime import datetime
tstamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# model_id = "unsloth/Meta-Llama-3.1-8B"
model_id = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# run poor person's hyperparameter sweep
a_learning_rate = [1e-4, 2e-4, 3e-4]
d_learning_rate = random.choice(a_learning_rate)
d_weight_decay = float(random.randrange(100) / 100)
a_scheduler = ['linear', 'constant', 'cosine']
s_scheduler = random.choice(a_scheduler)
i_warmup_steps = int(random.randrange(50))
a_r = [8, 16, 32, 64, 128]
i_r = random.choice(a_r)
a_lora_alpha = [8, 16, 32, 64, 128]
i_lora_alpha = random.choice(a_lora_alpha)
a_lora_dropout = [0, 0.1]
d_lora_dropout = random.choice(a_lora_dropout)
a_per_device_train_batch_size = [2, 4]
i_per_device_train_batch_size = random.choice(a_per_device_train_batch_size)
# a_gradient_accumulation_steps = [2, 4]
# i_gradient_accumulation_steps = random.choice(a_gradient_accumulation_steps)

# d_learning_rate = 2e-4
# d_weight_decay = .51
# s_scheduler = "constant"
# i_warmup_steps = 18
# i_r = 32
# i_lora_alpha = 64
# d_lora_dropout = 0.1

"""We now add LoRA adapters so we only need to update 1 to 10% of all parameters!"""
seed = 42
model = FastLanguageModel.get_peft_model(
    model,
    # r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    r = i_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    # lora_alpha = 16,
    # lora_alpha = 32,
    lora_alpha = i_lora_alpha,
    lora_dropout = d_lora_dropout, # Supports any, but = 0 is optimized
    # lora_dropout = 0.1, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = seed,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

alpaca_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
alpaca_prompt_no_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n{}"

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        if len(input) > 0:
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        else:
            text = alpaca_prompt_no_input.format(instruction, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

# grab dataset from the W&B Dataset Registry
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

train_dataset = ds["train"].map(formatting_prompts_func, batched = True,)
eval_dataset = ds["test"].map(formatting_prompts_func, batched = True,)

# print(str(eval_dataset))

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import evaluate

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

import numpy as np
import nltk
nltk.download('punkt_tab')


from transformers import GenerationConfig

gen_config = GenerationConfig.from_pretrained(model_id)
test_config = {
    "max_new_tokens" : 256,
    "gen_config" : gen_config
}

from tqdm.auto import tqdm
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
def generate(prompt, max_new_tokens=test_config["max_new_tokens"], gen_config=gen_config):
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
    with torch.inference_mode():
        output = model.generate(tokenized_prompt, 
                            max_new_tokens=max_new_tokens, 
                            pad_token_id=tokenizer.pad_token_id,
                            generation_config=gen_config)
    return tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

def prompt_table(examples, log=False, table_name="predictions"):
    table = wandb.Table(columns=["prompt", "generation", "concat", "output", "max_new_tokens", "temperature", "top_p"])
    # for example in tqdm(examples, leave=False):
    for example in examples.to_dict(orient="records"):
        prompt = alpaca_prompt_no_input.format(example["instruction"], "");
        if len(example["input"]) > 0:
            prompt = alpaca_prompt.format(example["instruction"], example["input"], "");
        prompt, gpt4_output = prompt, example["output"]
        # print(prompt + " -- " + gpt4_output)
        # prompt, gpt4_output = example["prompt"], example["output"]
        out = generate(prompt, test_config["max_new_tokens"], test_config["gen_config"])
        table.add_data(prompt, out, prompt+out, gpt4_output, test_config["max_new_tokens"], test_config["gen_config"].temperature, test_config["gen_config"].top_p)
    if log:
        wandb.log({table_name:table})
    return table

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

import pandas as pd
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

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        # per_device_train_batch_size = i_per_device_train_batch_size,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # consider 2,4
        warmup_steps = i_warmup_steps,
        num_train_epochs = 3,
        learning_rate = d_learning_rate,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = d_weight_decay,
        lr_scheduler_type = s_scheduler,
        seed = seed,
        output_dir = "outputs",
        report_to = "wandb",
        run_name = "llama-3.1-8b-fine-tune-" + tstamp,
        eval_strategy = "epoch",
        save_strategy="epoch",
    ),
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer_stats = trainer.train()

metrics_inference = compute_metrics_on_output()
wandb.log(metrics_inference)

run.finish()
