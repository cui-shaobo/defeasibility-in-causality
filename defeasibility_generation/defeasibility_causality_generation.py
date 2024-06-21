#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import wandb
import torch
import spacy
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Union
from datasets import Dataset, DatasetDict, load_metric
from torch.utils.data import DataLoader

from transformers import pipeline
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers.trainer_callback import PrinterCallback
from transformers import BartTokenizer, T5Tokenizer, GPT2Tokenizer
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration, GPT2LMHeadModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

# Disable warning
logging.disable(logging.WARNING)


def parse_args():
    """ Parse arguments.

    Returns:
        argparse.ArgumentParser: The arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate the supporter or defeater given the cause and the long term effect",
        usage="defeasibility_causality_generation.py [<args>] [-h | --help]"
    )

    parser.add_argument("--train_path", type=str, required=True,
                        help="the path of training set csv file")
    parser.add_argument("--val_path", type=str, required=True,
                        help="the path of validation set csv file")
    parser.add_argument("--test_path", type=str, required=True,
                        help="the path of test set csv file")
    parser.add_argument("-o", "--output_dir", type=str, default="./outputs",
                        help="the output directory of huggingface training")
    parser.add_argument("-p", "--predict_dir", type=str, default="./predictions.csv",
                        help="the predictions from the test set")
    parser.add_argument("-m", "--model", type=str, required=True, choices=["bart", "t5", "t5l", "gpt2"],
                        help="the model selected for training, only bart, t5-base, and t5-large is available")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="the number of random seeds for reproduction")
    parser.add_argument("-bsz", "--batch_size", type=int, default=16,
                        help="the batch size for training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5,
                        help="the learning rate for training")
    parser.add_argument("-e", "--epochs", type=int, default=3,
                        help="the epochs for training")
    parser.add_argument("-r", "--reason_type", type=str, required=True, choices=["supporter", "defeater"],
                        help="whether to generate supporter or defeater")
    parser.add_argument("--fp16", action="store_true",
                        help="whether to use fp16 (for t5-large)")
    parser.add_argument("-pp", "--parallelize", action="store_true",
                        help="whether to use parallel training")

    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproduction.

    Args:
        seed (int): A number for the seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_delta_causal(train_path, val_path, test_path, reason_type, seed=42):
    """Format the dataset to get causes, long term effects, and reasoning outputs (supporters or defeaters).

    Load and format the delta_causal dataset.
    Given the cause and the long term effects, generate the reasoning output (supporter or defeater).
    The long term effect will be the concatenation of time interval and the effect.
    It returns a huggingface dataset.

    Args:
        train_path (string): The csv data path of delta_causal training set.
        val_path (string): The csv data path of delta_causal validation set.
        test_path (string): The csv data path of delta_causal test set.
        reason_type (str): Whether to generate supporter or defeater.
        seed (int): Random seed for train, dev, test split.

    Returns:
        DatasetDict: Processed huggingface dataset with train, test, and dev set.
    """

    # Selected columns of the dataframe.
    sel_cols = [
        "cause",
        reason_type,
        "long_term_interval",
        "long_term_effect",
    ]

    # Read dataframe and select columns
    delta_causal_train = pd.read_csv(train_path)[sel_cols]
    delta_causal_val = pd.read_csv(val_path)[sel_cols]
    delta_causal_test = pd.read_csv(test_path)[sel_cols]


    # Concatenate the long term effect with the long term time interval
    # Convert to huggingface dataset
    def process_delta_causal(delta_causal):
        delta_causal["long_term_effect"] = delta_causal.long_term_interval + " "\
            + delta_causal.long_term_effect
        delta_causal.drop(
            columns=[
                "long_term_interval",
            ],
            inplace=True
        )
        delta_causal = Dataset.from_pandas(delta_causal)
        return delta_causal


    delta_causal_train = process_delta_causal(delta_causal_train)
    delta_causal_val = process_delta_causal(delta_causal_val)
    delta_causal_test = process_delta_causal(delta_causal_test)

    # Put all together
    train_test_valid_dataset = DatasetDict({
        'train': delta_causal_train,
        'test': delta_causal_test,
        'valid': delta_causal_val
    })

    return train_test_valid_dataset


def tokenize_dataset_for_seq2seq(examples, tokenizer, model="bart", reason_type="supporter"):
    """Tokenize the dataset to get the required input and output of huggingface models with seq2seq structures.

    Args:
        examples (DatasetDict): Raw delta_causal huggingface dataset.
        tokenizer: Huggingface tokenizer.
        model (str): Which model to use (t5 or bart), if t5 is selected, need to add a prefix.
        reason_type (str): Whether to generate supporter or defeater.

    Returns:
        DatasetDict: Tokenized dataset with input ids, attention masks, and labels.
    """

    # Add a prefix
    prefix = "" if model == "bart" else f"generate the {reason_type}: "

    model_inputs = tokenizer(
        [prefix+example for example in examples["cause"]],
        examples["long_term_effect"],
        max_length=128,
        truncation=True
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[reason_type], max_length=64, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_dataset_for_clm(examples, tokenizer, model, reason_type="supporter"):
    """Tokenize the dataset to get the required input and output of huggingface causal language models.

    Args:
        examples (DatasetDict): Raw delta_causal huggingface dataset.
        tokenizer: Huggingface tokenizer.
        model (str): Which model to use (only support gpt2).
        reason_type (str): Whether to generate supporter or defeater.

    Returns:
        DatasetDict: Tokenized dataset with input ids, attention masks, and labels.
    """

    assert model == "gpt2", "Only support GPT-2 models!"

    cause_sentences = ["cause: "+example for example in examples["cause"]]
    effect_sentences = ["long term effect: "+example for example in examples["long_term_effect"]]
    reason_sentences = [f"{reason_type}: "+example for example in examples[reason_type]]

    sentences = [
        f"{cause_sentences[i]} {effect_sentences[i]} {reason_sentences[i]}" for i, _ in enumerate(cause_sentences)
    ]

    output = tokenizer(
        sentences,
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    output["labels"] = output["input_ids"].copy()

    return output


def crop_sent(s, spacy_model, sent_idx=0):
    try:
        ss = list(spacy_model(s).sents)[sent_idx].text
    except:
        print(s)
        ss = s
    ss = ss.strip().replace('\n', ' ').strip()
    return ss


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    tqdm.pandas()

    # Load the dataframe
    dataset = load_delta_causal(args.train_path, args.val_path, args.test_path, args.reason_type, args.seed)

    # Load the tokenizer and the model
    if args.model == "bart":
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        tokenizer.pad_token = tokenizer.eos_token
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    elif args.model == "t5":
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
    elif args.model == "t5l":
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        model = T5ForConditionalGeneration.from_pretrained("t5-large")
    elif args.model == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained("gpt2", is_decoder=True)

    # Parallel training (for T5-large)
    if args.parallelize:
        device_map = {
            0: [0, 1, 2],
            1: [3, 4, 5, 6, 7, 8, 9],
            2: [10, 11, 12, 13, 14, 15, 16],
            3: [17, 18, 19, 20, 21, 22, 23],
        }
        model.parallelize(device_map)

    if args.model == "gpt2":
        tokenized_dataset = dataset.map(
            lambda f: tokenize_dataset_for_clm(f, tokenizer, args.model, args.reason_type),
            batched=True,
            num_proc=4,
            remove_columns=dataset["train"].column_names
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if args.fp16 else None,
        )
    else:
        tokenized_dataset = dataset.map(
            lambda f: tokenize_dataset_for_seq2seq(f, tokenizer, args.model, args.reason_type),
            batched=True,
            num_proc=4,
            remove_columns=dataset["train"].column_names
        )
        label_pad_token_id = -100
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if args.fp16 else None,
        )

    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["valid"]
    test_dataset = tokenized_dataset["test"]

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end = True,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.train()

    # Predict the result on the test set
    # Save the result to the csv format
    predictions = []
    prefix = "" if args.model == "bart" else f"generate the {args.reason_type}: "
    for idx, example in tqdm(enumerate(dataset["test"])):
        if args.model == "gpt2":
            cause_sentence = "cause: "+example["cause"]
            effect_sentence = "long term effect: "+example["long_term_effect"]
            reason_sentence = f"{args.reason_type}:"
            sentence = cause_sentence+" "+effect_sentence+" "+reason_sentence
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )["input_ids"].to("cuda")
        else:
            inputs = tokenizer(
                prefix+example["cause"],
                example["long_term_effect"],
                max_length=128,
                return_tensors="pt",
                truncation=True
            )["input_ids"].to("cuda")
        outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if args.model == "gpt2":
            answer = answer[answer.find(f"{args.reason_type}:")+len(f"{args.reason_type}:"):]
        predictions.append(answer)

    df_predictions = pd.DataFrame({
        "ground_truth": dataset["test"][args.reason_type],
        "prediction": predictions
    })
    spacy_model = spacy.load('en_core_web_md')
    df_predictions.prediction = df_predictions.prediction.apply(
        lambda s: crop_sent(s, spacy_model)
    )
    if trainer.is_world_process_zero():
        df_predictions.to_csv(args.predict_dir)
