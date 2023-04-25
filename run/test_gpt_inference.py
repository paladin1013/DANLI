import numpy as np
import json
from model.model.llm_subgoal_predictor import LLMSubgoalPredictor
from model.utils.data_util import process_edh_for_subgoal_prediction
import os
import random

import re
import torch
import pytorch_lightning as pl
from attrdict import AttrDict
from pprint import pprint
from transformers import AutoTokenizer
from model.data.subgoal_seq2seq import SubgoalDataset
from model.model.model_plm import PLModelWrapper
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
    
    test_case_num = 10
    random.seed(SEED)
    edh_files = os.listdir(EDH_JSON_ROOT)
    # edh_samples = random.sample(edh_files, test_case_num)
    edh_samples = edh_files
    logger = logging.getLogger("test_gpt_inference")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    subgoal_predictor_dir = os.path.join(os.environ["DANLI_MODEL_DIR"], "subgoal_predictor")
    subgoal_predictor_ckpt = "ckpt.pth"
    model_args = AttrDict(
        json.load(open(os.path.join(subgoal_predictor_dir, "args.json")))
    )
    model_args["data_dir"] = os.path.join(
        os.environ["DANLI_DATA_DIR"], "processed_20220610"
    )
    gpu_num = torch.cuda.device_count()

    pl.seed_everything(SEED)
    
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(subgoal_predictor_dir, "tokenizer/")
    )
    
    device = "cuda:0"
    logger.info("Create model...")
    model_ckpt = os.path.join(
        subgoal_predictor_dir, subgoal_predictor_ckpt
    )
    model_args.device = device
    model = PLModelWrapper.load_from_checkpoint(
        model_ckpt, args=model_args, tokenizer=tokenizer
    )
    model = model.to(device=device)
    model.eval()
    model.freeze()
    
    data_path = os.path.join(os.environ["DANLI_DATA_DIR"], "processed_20220610")
    raw_edh_path = os.path.join(
        os.environ["DANLI_DATA_DIR"], "edh_instances/valid_unseen"
    )
    
    
    
    llm_sg_predictor = LLMSubgoalPredictor()
    assert llm_sg_predictor.gpt_api.check_connectivity()
    
    dataset = SubgoalDataset(tokenizer, args=model_args, split=["valid_unseen"])
    with open("teach-dataset/processed_20220610/games_gt_subgoals.json") as f:
        gt_subgoals_all = json.load(f)

    result_data = []
    for edh_sample_file in tqdm.tqdm(edh_samples):
        sample_data = {}
        edh_raw, edh_text = edh_file_loader(edh_sample_file)


        
        logger.debug(f"\n================== Sample {edh_sample_file} ===================\n")
        
        logger.debug("\n=== Ground Truth ===\n")
        game_id, edh_num = re.match("(\w+?)\.edh(\w+?)\.json", edh_sample_file).groups()
        sample_data['game_id'] = game_id
        sample_data['edh_num'] = edh_num



        gt_subgoals_sample = gt_subgoals_all[game_id]
        if "edh_to_gt_subgoal_idx" not in gt_subgoals_sample.keys():
            logger.warning(f"Game {game_id} does not have valid groundtruth subgoals")
            continue
        idx = gt_subgoals_sample["edh_to_gt_subgoal_idx"][f"edh{edh_num}"]
        gt_subgoals = [sg for k, sg in enumerate(gt_subgoals_sample["gt_all_subgoals"]) if k in idx]
        logger.debug(gt_subgoals)
        sample_data['subgoals_groundtruth'] = gt_subgoals
        sample_data['subgoals_groundtruth_raw'] = edh_raw["future_subgoals"]


        logger.debug("\n=== Bard Predicted Subgoal ===\n")
        batch = dataset.data_collect(
            [edh_text], inference_mode=True, device=device
        )
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device=device)
        predictions = model.predict(batch, max_step=64)
        future_subgoals_raw = predictions[0]["subgoals_future"][:-1]
        future_subgoals = []
        for sg in future_subgoals_raw:
            if sg[1] != "parentReceptacles":
                future_subgoals.append((sg[0], sg[1], None))
            else:
                future_subgoals.append(sg)
        sample_data['subgoal_predicted'] = future_subgoals
        logger.debug(future_subgoals)

        logger.debug("\n=== GPT Prompt ===\n")
        edh_input = llm_sg_predictor.parse_edh_data(edh_raw, edh_text["text_dialog_and_act"])
        # llm_sg_predictions = llm_sg_predictor.predict(edh_input)
        prompt = llm_sg_predictor.gen_edh_prompt(edh_input)
        logger.debug(prompt)
        sample_data['text_dialog_and_act'] = edh_text["text_dialog_and_act"]
        sample_data['gpt_prompt'] = prompt
        
        # logger.debug("\n=== GPT3.5 Response ===\n")
        # replies = llm_sg_predictor.gpt_api.send(
        #     [prompt, llm_sg_predictor.gen_formatting_prompt()]
        # )
        # logger.debug(replies)
        # sample_data['gpt35_reply_raw'] = replies[0]
        # sample_data['gpt_reply_formatted'] = replies[0]

        result_data.append(sample_data)
        
    with open(f"{GPT_RESULT_ROOT}/prompts.json", "w") as f:
        json.dump(result_data, f)