import numpy as np
import json
from model.model.llm_subgoal_predictor import LLMSubgoalPredictor
from model.utils.data_util import process_edh_for_subgoal_prediction
import os
import random
import re
from model.utils.data_util import process_edh_for_subgoal_prediction
from typing import Dict
import logging
import tqdm
import sys
import jsonlines


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
    random.seed(SEED)
    edh_files = os.listdir(EDH_JSON_ROOT)
    # edh_samples = random.sample(edh_files, test_case_num)
    edh_samples = edh_files
    logger = logging.getLogger("test_gpt_inference")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    
    data_path = os.path.join(os.environ["DANLI_DATA_DIR"], "processed_20220610")
    raw_edh_path = os.path.join(
        os.environ["DANLI_DATA_DIR"], "edh_instances/valid_unseen"
    )
    
    llm_sg_predictor = LLMSubgoalPredictor()
    result_data = []
    for edh_sample_file in tqdm.tqdm(edh_samples):
        sample_data = {}
        edh_raw, edh_text = edh_file_loader(edh_sample_file)
        
        logger.debug(f"\n================== Sample {edh_sample_file} ===================\n")
        
        game_id, edh_num = re.match("(\w+?)\.edh(\w+?)\.json", edh_sample_file).groups()
        sample_data['id'] = f"{game_id}-edh{edh_num}"
        edh_input = llm_sg_predictor.parse_edh_data(edh_raw, edh_text["text_dialog_and_act"])
        # llm_sg_predictions = llm_sg_predictor.predict(edh_input)
        prompt = llm_sg_predictor.gen_edh_prompt(edh_input)
        sample_data['text'] = [prompt, llm_sg_predictor.gen_formatting_prompt()]
        sample_data['first_text_length'] = len(prompt)
        sample_data['all_answers'] = []
        logger.debug(prompt)
        # sample_data['text_dialog_and_act'] = edh_text["text_dialog_and_act"]
        # sample_data['gpt_prompt'] = prompt
        result_data.append(sample_data)
        
    with jsonlines.open(f"{GPT_RESULT_ROOT}/prompts_gpt4.jsonl", "w") as writer:
        writer.write_all(result_data)
