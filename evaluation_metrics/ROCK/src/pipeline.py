"""
pipeline.py
"""
import itertools, ast, re
import transformers
import spacy
import lemminflect
import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path
from itertools import accumulate

"""
srl processor
"""
class AllenSRLWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, text):
        return self.model.predict(text)
    def get_tokens(self, text):
        return self.model._tokenizer.tokenize(text)
    @classmethod
    def get_default_wrapper(cuda_device=0):
        import allennlp_models
        import allennlp_models.pretrained
        return AllenSRLWrapper(allennlp_models.pretrained.load_predictor("structured-prediction-srl-bert",
                                                                         cuda_device=cuda_device))


"""
temporal predictor
"""
class TempPredictor:
    def __init__(self, model, tokenizer, device,
                 spacy_model="en_core_web_sm"):
        self._model = model
        self._model.to(device)
        self._model.eval()
        self._tokenizer = tokenizer
        self._mtoken = self._tokenizer.mask_token
        self.unmasker = transformers.pipeline("fill-mask", model=self._model, tokenizer=self._tokenizer, device=0)
        try:
            self._spacy = spacy.load(spacy_model)
        except Exception as e:
            self._spacy = spacy.load("en_core_web_sm")
            print(f"Failed to load spacy model {spacy_model}, use default 'en_core_web_sm'\n{e}")

    def _extract_token_prob(self, arr, token, crop=1):
        for it in arr:
            if len(it["token_str"]) >= crop and (token == it["token_str"][crop:]):
                return it["score"]
        return 0.

    def _sent_lowercase(self, s):
        try:
            return s[0].lower() + s[1:]
        except:
            return s

    def _remove_punct(self, s):
        try:
            return s[:-1]
        except:
            return s

    def predict(self, e1, e2, top_k=5):
        """
        returns
        """
        txt = self._remove_punct(e1) + " " + self._mtoken + " " + self._sent_lowercase(e2)
#         print(txt)
        return self.unmasker(txt, top_k=top_k)

    def batch_predict(self, instances, top_k=5, batch_size=8):
        txt = [self._remove_punct(e1) + " " + self._mtoken + " " + self._sent_lowercase(e2)
                for (e1, e2) in  instances]
        return self.unmasker(txt, top_k=top_k, batch_size=batch_size)


    def get_temp(self, e1, e2, top_k=5, crop=1):
        inst1 = self.predict(e1, e2, top_k)
        inst2 = self.predict(e2, e1, top_k)

        # e1 before e2
        b1 = self._extract_token_prob(inst1, "before", crop=crop)
        b2 = self._extract_token_prob(inst2, "after", crop=crop)

        # e1 after e2
        a1 = self._extract_token_prob(inst1, "after", crop=crop)
        a2 = self._extract_token_prob(inst2, "before", crop=crop)

        return (b1+b2)/2, (a1+a2)/2

    def get_temp_batch(self, instances, top_k=5, crop=1, batch_size=8):
        reverse_instances = [(e2, e1) for (e1, e2) in instances]
        fwd_preds = self.batch_predict(instances, top_k=top_k, batch_size=batch_size)
        bwd_preds = self.batch_predict(reverse_instances, top_k=top_k, batch_size=batch_size)

        b1s = np.array([ self._extract_token_prob(pred, "before", crop=crop) for pred in fwd_preds ])
        b2s = np.array([ self._extract_token_prob(pred, "before", crop=crop) for pred in bwd_preds ])
        a1s = np.array([ self._extract_token_prob(pred, "after", crop=crop) for pred in fwd_preds ])
        a2s = np.array([ self._extract_token_prob(pred, "after", crop=crop) for pred in bwd_preds ])

        return np.array([np.array(b1s+b2s)/2, np.array(a1s+a2s)/2]).T


    def __call__(self, *args, **kwargs):
        return self.get_temp(*args, **kwargs)


"""
polyjuice intervention
"""
class PJGenerator:
    PERETURB_TOK = "<|perturb|>"
    BLANK_TOK = "[BLANK]"
    SEP_TOK = "[SEP]"
    EMPTY_TOK = "[EMPTY]"
    ANSWER_TOK = "[ANSWER]"

    def __init__(self, model_path="uw-hai/polyjuice",
                 spacy_processor=None,
                 device=0,
                 srl_processor=None,
                ):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        generator = transformers.pipeline("text-generation",
            model=AutoModelForCausalLM.from_pretrained(model_path),
            tokenizer=AutoTokenizer.from_pretrained(model_path),
            framework="pt", device=0)
        generator.tokenizer.pad_token = generator.tokenizer.eos_token
        self.generator = generator

        if spacy_processor is None:
            spacy_processor = spacy.load('en_core_web_md')
        self.spacy_processor = spacy_processor

        if srl_processor is None:
            srl_processor = AllenSRLWrapper.get_default_wrapper(cuda_device=0 if torch.cuda.is_available() else None)
        self.srl_processor = srl_processor
    def get_prompts(self, doc, ctrl_codes, blanked_sents, is_complete_blank=True):
        prompts = []
        for tag, bt in itertools.product(ctrl_codes, blanked_sents):
            sep_tok = self.SEP_TOK if bt and is_complete_blank else ""
            prompts.append(f"{doc.text.strip()} {self.PERETURB_TOK} [{tag}] {bt.strip()} {sep_tok}".strip())
        return prompts

    def generate_on_prompts(self, generator, prompts, clean_output=True, temperature=1,
        num_beams=None, n=3, do_sample=True, batch_size=128, **kwargs):

        def split_ctrl_code(text):
            r = re.search(r'\[(?P<code>[a-z]+)\](?P<text>.+)', text)
            if r:
                return r.group("code").strip(), r.group("text").strip()
            return "", text

        def remove_blanks(text):
            try:
                before, answers = text.split(self.SEP_TOK)
                answers = [x.strip() for x in answers.split(self.ANSWER_TOK)][:-1]
                answers = [x if x != self.EMPTY_TOK else '' for x in answers]
                for a in answers:
                    if a == '':
                        before = re.sub(r' %s' % re.escape(self.BLANK_TOK), a, before, count=1)
                    else:
                        before = re.sub(r'%s' % re.escape(self.BLANK_TOK), a, before, count=1)
                return before, answers
            except:
                return text, []

        def batched_generate(generator,
                examples,
                temperature=1,
                num_beams=None,
                num_return_sequences=1,
                do_sample=True,
                batch_size=128, **kwargs):
            preds_list = []
            with torch.no_grad():
                for e in (range(0, len(examples), batch_size)):
                    preds_list += generator(
                        examples[e:e+batch_size],
                        temperature=temperature,
                        #return_tensors=True,
                        num_beams=num_beams,
                        max_length=1000,
                        early_stopping=None if num_beams is None else True,
                        do_sample=num_beams is None and do_sample,
                        num_return_sequences=num_return_sequences,
                        pad_token_id=self.generator.tokenizer.eos_token_id,
                        **kwargs
                    )
            return preds_list

        preds_list = batched_generate(generator, prompts,
            temperature=temperature, n=n,
            num_beams=num_beams,
            do_sample=do_sample, batch_size=batch_size, **kwargs)

        if len(prompts) == 1:
            preds_list = [preds_list]

#         if clean_output:
        preds_list_cleaned = []
        for prompt, preds in zip(prompts, preds_list):
            prev_list = set()
            for s in preds:
                total_sequence = s["generated_text"].split(self.PERETURB_TOK)[-1]
                normalized, _ = remove_blanks(total_sequence)
                input_ctrl_code, normalized = split_ctrl_code(normalized)
                prev_list.add(normalized)
            preds_list_cleaned.append(list(prev_list))
        return preds_list_cleaned
#         return preds_list


    def generate_blanks_via_srl(self, text):
        def proper_whitespaces(text):
            return re.sub(r'\s([?.!"](?:\s|$))', r'\1', text.strip())
        srl = self.srl_processor(text)
        tokens = srl['words']
        verbs = srl['verbs']
        tgts = ['ARG0', 'V', 'ARG1']
        blanks = []
        for v in verbs:
            tags = v['tags']
            for tgt in tgts:
                if f"B-{tgt}" not in tags:
                    continue
                blk_start = tags.index(f"B-{tgt}")
                blk_end = blk_start+1 if f"I-{tgt}" not in tags else len(tags)-tags[::-1].index(f"I-{tgt}")
                sent = ' '.join(tokens[:blk_start]) + " [BLANK] " + ' '.join(tokens[blk_end:])
                blanks.append(proper_whitespaces(sent))
        return blanks

    def agg_generations(self, gen):
        agg = {}
        for lists in gen:
            for (ctrl, sent) in lists:
                if ctrl not in agg:
                    agg[ctrl] = []
                agg[ctrl].append(sent)
        return agg

    def __call__(self, sentences,
                 ctrl_codes=["resemantic", "negation", "lexical"],
                 aggregation=True,
                 seed=42,
                 **kwargs):
        
        # [sent1, sent2, ...]
        doc_list = [self.spacy_processor(text) for text in sentences]

        # [[sent1 blank1, sent1 blank2, ...], [sent2 blank1, ...]] avg size * 4?
        blanks_list = [self.generate_blanks_via_srl(text) for text in sentences]

        # for each sent_i blank_j, generate 8 prompts according to different sentiments
        prompts = [self.get_prompts(doc, ctrl_codes, blanks) for (doc, blanks) in zip(doc_list, blanks_list)]

        random.seed(seed)
        # for each sent_i, only sample 10 of them.
        sampled_prompts = [random.sample(prompt, min(10, len(prompt))) for prompt in prompts]
        
        # flatten the list and keep the start index of each sentence
        flattened_prompts = sum(sampled_prompts, [])
        index_corresponding_to_prompts = [len(prompt) for prompt in sampled_prompts]
        index_corresponding_to_prompts = list(np.subtract(list(accumulate(index_corresponding_to_prompts)), index_corresponding_to_prompts))
        index_corresponding_to_prompts.append(len(flattened_prompts))

        generations = self.generate_on_prompts(generator=self.generator, prompts=flattened_prompts, **kwargs)
        generations_list = []

        for idx in range(len(index_corresponding_to_prompts)-1):
            st = index_corresponding_to_prompts[idx]
            ed = index_corresponding_to_prompts[idx+1]
            generations_list.append(sum(generations[st:ed], []))

        return generations_list
