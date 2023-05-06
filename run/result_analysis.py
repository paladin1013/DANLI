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

os.chdir(os.environ["DANLI_ROOT_DIR"])
pred = pd.read_json(f"{GPT_RESULT_ROOT}/{PRED}")

with open(f"{GPT_RESULT_ROOT}/{GPT_OUTPUT}", "r") as f:
    gpt_out_lines = list(f)

gpt_out_list = []
for line in gpt_out_lines:
    gpt_out_list.append(json.loads(line))
gpt_out = pd.DataFrame.from_dict(gpt_out_list)

# %%
def get_respond(df: pd.DataFrame, game_id:str, edh_num:int):
    series = df[df['id'] ==f"{game_id}-edh{edh_num}"]
    return series["all_answers"].to_list()[0][1]

# print(get_respond(gpt_out, "bfa7505b440eadde_7fab", 2))

# %% Parse results
llm_sg_predictor = LLMSubgoalPredictor()


def process_series(series):
    id_str:str = series['id']
    series['game_id'] = id_str.split('-edh')[0]
    series['edh_num'] = int(id_str.split('-edh')[1])
    if pd.isna(series['error']):
        try:
            series['results'] = llm_sg_predictor.parse_gpt_reply(series["all_answers"][1])
        except ValueError as e:
            print(f"Parsing results failed for game {series['game_id']} edh{series['edh_num']}")
            print(f"  {str(e)}")
            # print(f"  gpt response: {series['all_answers'][1]}")
            
    return series
    
    
gpt_out = gpt_out.apply(process_series, axis=1)

