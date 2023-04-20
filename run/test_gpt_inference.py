import numpy as np
import json
from model.model.llm_subgoal_predictor import LLMSubgoalPredictor
from model.utils.data_util import process_edh_for_subgoal_prediction


SPLIT = "valid_seen"
EDH_JSON_ROOT = f"teach-dataset/edh_instances/{SPLIT}/"
PROCESSED_SUBGOAL_FILE = f"teach-dataset/processed_20220610/seq2seq_sg_pred_{SPLIT}.json"

def edh_file_loader(game_id:str, edh_id:int):
    
    with open(f"{EDH_JSON_ROOT}/{game_id}.edh{edh_id}.json") as f:
        edh_raw = json.load(f)

    edh_text, dialog_history = process_edh_for_subgoal_prediction(edh_raw)
    return edh_raw, edh_text["text_dialog_and_act"]
        

if __name__ == "__main__":
    llm_sg_predictor = LLMSubgoalPredictor()
    
    edh_raw, text_dialog_and_act = edh_file_loader("c062b518366ad411", 0)
    edh_input = llm_sg_predictor.parse_edh_data(edh_raw, text_dialog_and_act)
    llm_sg_predictions = llm_sg_predictor.predict(edh_input)
    print(llm_sg_predictions)