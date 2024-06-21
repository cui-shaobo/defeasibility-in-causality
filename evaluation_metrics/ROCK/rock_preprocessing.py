#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import torch
import random
import argparse
import nltk, spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging, warnings
import transformers, allennlp
from transformers import (AutoTokenizer, AutoModelForMaskedLM,
                          RobertaModel,RobertaForMaskedLM, 
                          RobertaTokenizer, GPT2LMHeadModel, GPT2Tokenizer)
import allennlp_models
import allennlp_models.pretrained
from transformers import AutoTokenizer, AutoModelForCausalLM

import src
import src.pipeline
import src.utils as utils


def parse_args():
    """ Parse arguments.

    Returns:
        argparse.ArgumentParser: The arguments.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess the delta_causal dataset to use the ROCK metric.",
        usage="rock_preprocessing.py [<args>] [-h | --help]"
    )

    parser.add_argument("--data_path", type=str, required=True, help="the path of delta_causal csv file")
    parser.add_argument("--output_path", type=str, default="./delta_causal_processed.pkl", help="the preprocessed delta_causal dataframe saved with the pkl format")
    parser.add_argument("--seed", type=int, default=42, help="the number of random seeds for reproduction")
    parser.add_argument("--cov_bsz", type=int, default=8, help="the batch size for covariates generation")
    parser.add_argument("--itv_bsz", type=int, default=8, help="the batch size for interventions generation")
    parser.add_argument("--tmp_bsz", type=int, default=16, help="the batch size for temporal probability generation")

    return parser.parse_args()


def disable_warning():
    """Disable the warning messages.
    """
    logging.getLogger('allennlp.common.params').disabled = True 
    logging.getLogger('allennlp.nn.initializers').disabled = True 
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO) 
    logging.getLogger('urllib3.connectionpool').disabled = True 
    logging.getLogger().setLevel(logging.CRITICAL)
    warnings.filterwarnings('ignore')
    logging.disable(sys.maxsize)
    logging.disable(logging.WARNING)


def set_seed(seed):
    """Set random seeds for reproduction.

    Args:
        seed (int): A number for the seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_conan(data_path):
    """Format the dataset to get the 3 type of causes and long term effects.

    Load and format the t-conan dataset. We want to measure the causality strength between 3 different types of causes and the long-term effect. We use the cause itself, the cause concatenated with the assumption, the cause concatenated with the defeater. It returns a pandas dataframe.

    Args:
        data_path (string): The csv data path of delta_causal dataset.

    Returns:
        pandas.DataFrame: The processed delta_causal dataset in the format of pandas DataFrame.
    """

    # Read the dataframe and select columns
    sel_cols = [
        "cause",
        "long_term_interval",
        "long_term_effect",
        "defeater_time_interval",
        "defeater",
        "assumption"
    ]
    delta_causal = pd.read_csv(data_path)[sel_cols]

    # Process dataframes
    delta_causal["d_cause"] = delta_causal.cause\
        + " And then, "\
        + delta_causal.defeater
    delta_causal["a_cause"] = delta_causal.cause\
        + " And then, "\
        + delta_causal.assumption
    delta_causal["effect"] = delta_causal.long_term_effect

    # Remove the other columns
    delta_causal.drop(
        columns=[
            "long_term_interval",
            "long_term_effect",
            "defeater_time_interval",
            "defeater",
            "assumption"
        ],
        inplace=True
    )

    return delta_causal


def sample_cov(df, model, tokenizer, cov_bsz):
    """ Generate the covariates for each type of cause.

    Args:
        df (pandas.DataFrame): The loaded delta_causal dataframe.
        model: The huggingface model.
        tokenizer: The huggingface tokenizer.
        cov_bsz (int): The batch size for covariates generation.

    Returns:
        pandas.DataFrame: DataFrame with 3 types of covariates for 3 types of different causes (cause/cause+assumption/cause+defeater), marked with no prefix/"d_"/"a_".
    """
    for prefix in ["", "d_", "a_"]:
        txt_list = []
        for idx in tqdm(range(0, len(df), cov_bsz)):
            prompts = list(df.iloc[idx:min(idx+cov_bsz, len(df))][prefix+"cause"] + " Before that, ")
            gen_tokens = model.generate(
                tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True
                ).input_ids.cuda(), 
                do_sample=True,
                temperature=0.9,
                max_length=64,
                num_return_sequences=30,
                pad_token_id=tokenizer.eos_token_id,
            )
            txt = gpt_tokenizer.batch_decode(
                gen_tokens,
                skip_special_tokens=True
            )
            txt_list += [txt[idx: idx+30] for idx in range(0, len(txt), 30)]
        df[prefix+"covariates"] = txt_list

    return df


def sample_interventions(df, cf_gen, ctrl_codes, itv_bsz):
    """ Generate the interventions for each type of cause.

    Args:
        df (pandas.DataFrame): The loaded delta_causal dataframe.
        cf_gen: The intervention generation model.
        ctrl_codes (list of str): Types of the interventions.
        itv_bsz (int): The batch size for interventions generation.

    Returns:
        pandas.DataFrame: DataFrame with 3 types of interventions for 3 types of different causes (cause/cause+assumption/cause+defeater), marked with no prefix/"d_"/"a_".
    """
    for prefix in ["", "d_", "a_"]:
        txt_list = []
        for idx in tqdm(range(0, len(df), itv_bsz)):
            prompts = list(df.iloc[idx:min(idx+itv_bsz, len(df))][prefix+"cause"])
            cf_texts = cf_gen(
                prompts,
                ctrl_codes=ctrl_codes
            )
            txt_list += cf_texts
        df[prefix+"interventions"] = txt_list

    return df


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    tqdm.pandas()

    # Load the dataframe
    delta_causal = load_conan(args.data_path).iloc[0:250]

    # Load models for preprocessing
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt_model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    gpt_tokenizer.padding_side = "left"

    spacy_model = spacy.load('en_core_web_md')
    allensrl = src.pipeline.AllenSRLWrapper(allennlp_models.pretrained.load_predictor("structured-prediction-srl-bert", cuda_device=0))
    cf_gen = src.pipeline.PJGenerator(srl_processor=allensrl)

    tp_roberta_ft = src.pipeline.TempPredictor(
        model=RobertaForMaskedLM.from_pretrained("CogComp/roberta-temporal-predictor"),
        tokenizer=RobertaTokenizer.from_pretrained("CogComp/roberta-temporal-predictor"),
        device=0
    )

    # Sample covariates
    # Use the GPT2 model instead of GPT-J-6B because it is too big
    delta_causal = sample_cov(delta_causal, gpt_model, gpt_tokenizer, args.cov_bsz)

    # Generate interventions
    delta_causal = sample_interventions(
        delta_causal, cf_gen,
        [
            "resemantic", 
            "negation", 
            "lexical",
            "quantifier",
            "insert",
            "restructure",
            "shuffle",
            "delete"
        ],
        args.itv_bsz
    )

    # Compute the temporal probabilities
    delta_causal = delta_causal.progress_apply(
        lambda s : utils.conan_get_probs(s, tp_roberta_ft, spacy_model),
        axis=1
    )
    print(delta_causal)

    # Save to pickle
    delta_causal.to_pickle(args.output_path)
