# %% 
import json
import jsonlines
import pandas as pd
import os
from model.model.llm_subgoal_predictor import LLMSubgoalPredictor

# %% Merge data from bart predictions and gpt results

GPT_RESULT_ROOT = "teach-dataset/gpt_data"
PRED = "prompts.json"
GPT_OUTPUT = "gpt4_output.jsonl"
MERGED_OUTPUT = "merged_output.json"

os.chdir(os.environ["DANLI_ROOT_DIR"])
pred = pd.read_json(f"{GPT_RESULT_ROOT}/{PRED}")
pred = pred.rename(columns={'subgoal_predicted':'subgoals_bart'})

with open(f"{GPT_RESULT_ROOT}/{GPT_OUTPUT}", "r") as f:
    gpt_out_lines = list(f)

gpt_out_list = []
for line in gpt_out_lines:
    gpt_out_list.append(json.loads(line))

# # %%  
# def get_respond(df: pd.DataFrame, game_id:str, edh_num:int):
#     series = df[df['id'] ==f"{game_id}-edh{edh_num}"]
#     return series["all_answers"].to_list()[0][1]

# # print(get_respond(gpt_out, "bfa7505b440eadde_7fab", 2))

# %% Parse results
gpt_out = pd.DataFrame.from_dict(gpt_out_list)

llm_sg_predictor = LLMSubgoalPredictor(ignore_invalid=True)


def process_series(series):
    id_str:str = series['id']
    series['game_id'] = id_str.split('-edh')[0]
    series['edh_num'] = int(id_str.split('-edh')[1])
    if pd.isna(series['error']):
        series['subgoals_gpt4'], series['parse_error'] = llm_sg_predictor.parse_gpt_reply_to_str(series["all_answers"][1])
        # print(f"  gpt response: {series['all_answers'][1]}")
            
    return series
    
    
gpt_out = gpt_out.apply(process_series, axis=1).rename(columns={"all_answers": "gpt_reply_raw"})

pred_merged = pred.merge(gpt_out[['game_id', 'edh_num', 'parse_error', 'subgoals_gpt4', 'gpt_reply_raw']], on=['game_id', 'edh_num'])
pred_merged = pred_merged.reindex(columns=['game_id', 'edh_num', 'subgoals_groundtruth', 'subgoals_bart', 'subgoals_gpt4', 'text_dialog_and_act', 'gpt_prompt', 'gpt_reply_raw', 'parse_error'])
pred_merged.to_json(f"{GPT_RESULT_ROOT}/{MERGED_OUTPUT}", orient='records')