import numpy as np
import json
from model.model.llm_subgoal_predictor import LLMSubgoalPredictor
from model.utils.data_util import process_edh_for_subgoal_prediction
import os
import random
import re
from model.utils.format_utils import parse_edh_data
from typing import Dict
import logging
from tqdm import tqdm
import sys
import jsonlines
from typing import List, Dict
from model.model.gpt_api import GPTAPI
from model.data.memory_manager import TaskMemoryManager
SPLIT = "valid_unseen"
EDH_JSON_ROOT = f"teach-dataset/edh_instances/{SPLIT}/"
PROCESSED_SUBGOAL_FILE = f"teach-dataset/processed_20220610/seq2seq_sg_pred_{SPLIT}.json"
GPT_RESULT_ROOT = f"teach-dataset/gpt_data/"
SEED = 1234


def edh_file_loader(file_name: str):
    
    with open(f"{EDH_JSON_ROOT}/{file_name}") as f:
        edh_raw = json.load(f)

    edh_text, dialog_history = process_edh_for_subgoal_prediction(edh_raw)
    return edh_raw, edh_text


if __name__ == "__main__":
    edh_samples = os.listdir(EDH_JSON_ROOT)
    gpt_api = GPTAPI(model="gpt4")
    logger = logging.getLogger("generate_gpt_prompts")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    # llm_sg_predictor = LLMSubgoalPredictor()
    # sessions:List[List[str]] = []
    # ids:List[str] = []
    # for edh_sample_file in tqdm(edh_samples):
    #     sample_data = {}
    #     edh_raw, edh_text = edh_file_loader(edh_sample_file)
        
    #     logger.debug(f"\n================== Sample {edh_sample_file} ===================\n")
        
    #     game_id, edh_num = re.match("(\w+?)\.edh(\w+?)\.json", edh_sample_file).groups()
    #     ids.append(f"{game_id}-edh{edh_num}")
    #     edh_input = parse_edh_data(edh_raw, edh_text["text_dialog_and_act"])
    #     prompt = llm_sg_predictor.gen_edh_prompt(edh_input)
    #     sessions.append([prompt])
    #     logger.debug(prompt)
    
    # gpt_api.generate_jsonl(f"{GPT_RESULT_ROOT}/gpt4_{SPLIT}.jsonl", sessions, ids)

    for split in ["train", "valid_seen", "valid_unseen"]:
        manager = TaskMemoryManager(memory_split=split, data_root_dir="teach-dataset")
        manager.load_memory()
        sessions:Dict[str, Dict] = {}
        for task in manager.task_memory:
            sessions[task["game_id"]] = {"id": task["game_id"], "prompts":[manager.generate_explaination_prompt(task)]}
        gpt_api.generate_jsonl(f"{GPT_RESULT_ROOT}/prompts/subgoal_explainations_prompt_{split}.jsonl", sessions)
            