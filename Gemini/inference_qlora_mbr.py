#!/usr/bin/env python3
import os
import re
import warnings
import pandas as pd
import torch
from pathlib import Path
from typing import List
from tqdm.auto import tqdm
import sacrebleu

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

warnings.filterwarnings('ignore')

class MBR_QLoRA_Inference:
    def __init__(self, base_model_path, adapter_path, device="cuda"):
        self.device = device
        self.batch_size = 4
        self.max_length = 512
        self.max_new_tokens = 256
        
        # MBR Knobs
        self.use_mbr = True
        self.mbr_num_beam_cands = 4
        self.mbr_num_sample_cands = 4
        self.mbr_top_p = 0.92
        self.mbr_temperature = 0.8
        
        print(f"Loading Base Model: {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
        
        print(f"Loading Adapter: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = self.model.to(self.device).eval()
        
        self._CHRFPP_SENT = sacrebleu.metrics.CHRF(word_order=2)
        
        # Create Data pre/post processors
        self._init_postprocess_patterns()

    def _init_postprocess_patterns(self):
        self.fracts = {
            r'\.5\b': '½', r'\.25\b': '¼', r'\.75\b': '¾',
            r'\.33+\d*\b': '⅓', r'\.66+\d*\b': '⅔'
        }
        self.bad_chars = '!?()—–<>⌈⌋⌊[]+/'
        self.bad_trans = str.maketrans('', '', self.bad_chars)

    def preprocess(self, text):
        if pd.isna(text): return ""
        text = str(text)
        text = re.sub(r'(\[\s*x\s*\]|\(\s*x\s*\)|\bx\b)', '<gap>', text, flags=re.I)
        text = re.sub(r'(\.{3,}|…+)','<gap>', text)
        return "translate Akkadian to English: " + text.strip()

    def postprocess(self, text):
        if not text: return ""
        s = str(text)
        s = re.sub(r'\((fem|plur|pl|sing|singular|plural|\?|!)\.?\s*\w*\)', '', s, flags=re.I)
        
        s = s.replace('<gap>', '\x00GAP\x00')
        s = s.translate(self.bad_trans)
        s = s.replace('\x00GAP\x00', ' <gap> ')
        
        for pat, rep in self.fracts.items():
            s = re.sub(r'(\d+)' + pat, r'\1' + rep, s)
            s = re.sub(r'\b0' + pat, rep, s)
            
        return re.sub(r'\s+', ' ', s).strip()

    def _sim_chrfpp(self, a, b):
        if not a or not b: return 0.0
        return float(self._CHRFPP_SENT.sentence_score(a, [b]).score)

    def _mbr_pick(self, cands: List[str]) -> str:
        # Dedup
        seen = set()
        uniq = []
        for x in cands:
            x = x.strip()
            if x and x not in seen:
                uniq.append(x)
                seen.add(x)
                
        n = len(uniq)
        if n == 0: return ""
        if n == 1: return uniq[0]
        
        best_s, best_i = -1e9, 0
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i != j:
                    s += self._sim_chrfpp(uniq[i], uniq[j])
            s /= max(1, n - 1)
            if s > best_s:
                best_s, best_i = s, i
        return uniq[best_i]

    def _generate_mbr_batch(self, input_ids, attention_mask):
        B = int(input_ids.shape[0])
        gen_common = {"max_new_tokens": self.max_new_tokens, "use_cache": True}
        
        # Beams
        nb_cands = self.mbr_num_beam_cands
        beam_out = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            do_sample=False, num_beams=nb_cands, num_return_sequences=nb_cands,
            **gen_common
        )
        beam_txt = self.tokenizer.batch_decode(beam_out, skip_special_tokens=True)
        
        pools = [[] for _ in range(B)]
        for i in range(B):
            pools[i].extend(beam_txt[i * nb_cands:(i + 1) * nb_cands])
            
        # Samples (High Temp)
        ns_cands = self.mbr_num_sample_cands
        if ns_cands > 0:
            samp_out = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                do_sample=True, top_p=self.mbr_top_p, temperature=self.mbr_temperature,
                num_return_sequences=ns_cands, num_beams=1,
                **gen_common
            )
            samp_txt = self.tokenizer.batch_decode(samp_out, skip_special_tokens=True)
            for i in range(B):
                pools[i].extend(samp_txt[i * ns_cands:(i + 1) * ns_cands])
                
        return [self._mbr_pick(p) for p in pools]

    def run(self, df):
        print(f"Running MBR Inference on {len(df)} rows")
        results = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in tqdm(range(0, len(df), self.batch_size)):
                batch = df.iloc[i:i+self.batch_size]
                ids = batch["id"].tolist()
                texts = [self.preprocess(t) for t in batch["transliteration"]]
                
                toks = self.tokenizer(texts, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
                input_ids = toks.input_ids.to(self.device)
                att_mask = toks.attention_mask.to(self.device)
                
                chosen_raw = self._generate_mbr_batch(input_ids, att_mask)
                chosen_clean = [self.postprocess(t) for t in chosen_raw]
                
                for idx, t in zip(ids, chosen_clean):
                    results.append({"id": idx, "translation": t})
                    
        return pd.DataFrame(results)

if __name__ == "__main__":
    BASE_MODEL = "c:/Users/29421/Desktop/kaggle_Challenge/models/byt5-xl-akkadian"
    ADAPTER = "c:/Users/29421/Desktop/kaggle_Challenge/qlora/qlora_adapter_final_v2"
    TEST_CSV = "c:/Users/29421/Desktop/kaggle_Challenge/data/competition/test.csv"
    
    # We only run this if the adapter exists (i.e. model finished training)
    if os.path.exists(ADAPTER):
        engine = MBR_QLoRA_Inference(BASE_MODEL, ADAPTER)
        test_df = pd.read_csv(TEST_CSV)
        sub_df = engine.run(test_df)
        sub_df.to_csv("c:/Users/29421/Desktop/kaggle_Challenge/Gemini/submission_mbr.csv", index=False)
        print("Submission saved to Gemini/submission_mbr.csv")
    else:
        print(f"Adapter not yet available at {ADAPTER}. Wait for training to finish.")
