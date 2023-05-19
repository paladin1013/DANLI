# %% 
import json
import jsonlines
import pandas as pd
import os
import sys
from model.model.llm_subgoal_predictor import LLMSubgoalPredictor
from model.model.gpt_api import GPTAPI
import logging
from typing import List
import numpy as np
class DataAnalyzer:
    def __init__(self, result_root="teach-dataset/gpt_data"):
        self.result_root = result_root
        os.chdir(os.environ["DANLI_ROOT_DIR"])
        self.logger = logging.getLogger("DataAnalyzer")
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.setLevel(logging.DEBUG)
    def load_bart_data(self, bart_results="bart_result.json"):
        try:
            self.bart_results = pd.read_json(f"{self.result_root}/{bart_results}").rename(columns={'subgoal_predicted':'subgoals_bart'})
            self.results = self.bart_results
        except:
            self.logger.warning(f"Results of bart predictor {self.result_root}/{bart_results} does not exist!")

    def load_gpt_data(self, gpt_results="gpt4_output.jsonl"):
        gpt_api = GPTAPI(model="gpt4")
        sessions = gpt_api.load_response_jsonl(f"{self.result_root}/{gpt_results}")
        # try:
        gpt_out = pd.DataFrame.from_dict(sessions.values())
        self.gpt_results = self.parse_gpt_output(gpt_out)
        # except:
        #     self.logger.warning(f"Results of gpt predictor {self.result_root}/{gpt_results} does not exist!")

    
    def merge_results(self, merged_output="merged_output.json"):
        pred_merged = self.bart_results.merge(self.gpt_results[['game_id', 'edh_num', 'parse_error', 'subgoals_gpt4', 'gpt_reply_raw']], on=['game_id', 'edh_num'])
        pred_merged = pred_merged.reindex(columns=['game_id', 'edh_num', 'subgoals_groundtruth', 'subgoals_bart', 'subgoals_gpt4', 'text_dialog_and_act', 'gpt_prompt', 'gpt_reply_raw', 'parse_error'])
        pred_merged.to_json(f"{self.result_root}/{merged_output}", orient='records')
        self.results = pred_merged
        
    def get_respond(self, df: pd.DataFrame, game_id:str, edh_num:int):
        series = df[df['id'] ==f"{game_id}-edh{edh_num}"]
        return series["responses"].to_list()[0][1]

    # print(get_respond(gpt_out, "bfa7505b440eadde_7fab", 2))

    
    def parse_gpt_output(self, gpt_out:pd.DataFrame):
        llm_sg_predictor = LLMSubgoalPredictor(ignore_invalid=True)
        def process_series(series):
            id_str:str = series['id']
            series['game_id'] = id_str.split('-edh')[0]
            series['edh_num'] = int(id_str.split('-edh')[1])
            if pd.isna(series['error']):
                series['subgoals_gpt4'], series['parse_error'] = llm_sg_predictor.parse_gpt_reply_to_str(series["responses"][1])
                # print(f"  gpt response: {series['responses'][1]}")
            return series
        self.gpt_results = gpt_out.apply(process_series, axis=1).rename(columns={"responses": "gpt_reply_raw"})
        
    
    def calc_accuracy(self, predictor="bart"):
        
        def get_edh_session_acc(series):
            valid_nums = [0, 0, 0] # Corresponds to subject, predicate, object
            correct_nums = [0, 0, 0]
            gt = series['subgoals_groundtruth']
            pred = series[f'subgoals_{predictor}']
            # isPickedUp is not a independent subgoal. It can be planned when completing other subgoals
            pred = [pred_subgoal for pred_subgoal in pred if pred_subgoal[1] != "isPickedUp"]
            gt = [gt_subgoal for gt_subgoal in gt if gt_subgoal[1] != "isPickedUp"]

            # For simplicity, we calculate the first k subgoals (k is the number of groundtruth)
            for id, gt_subgoal in enumerate(gt):
                valid_nums[0] += 1
                valid_nums[1] += 1
                if id < len(pred):
                    pred_subgoal = pred[id]
                    if pred_subgoal[0] == gt_subgoal[0]:
                        correct_nums[0] += 1
                    if pred_subgoal[1] == gt_subgoal[1]:
                        correct_nums[1] += 1
                    if gt_subgoal[1] == "parentReceptacles":
                        valid_nums[2] += 1 # parentReceptacles is the only predicate that takes two arguments
                        if pred_subgoal[2] == gt_subgoal[2]:
                            correct_nums[2] += 1
                else:
                    if gt_subgoal[1] == "parentReceptacles":
                        valid_nums[2] += 1
            series[f'{predictor}_subgoal_statistics'] = [valid_nums, correct_nums]

            if valid_nums[0] == 0:
                # If no valid subgoals, we set the correctness to -1
                series[f'{predictor}_subgoal_correct'] = -1
            else:
                series[f'{predictor}_subgoal_correct'] = (valid_nums == correct_nums)

            return series
        
        self.results = self.results.apply(get_edh_session_acc, axis=1)
        correctnesses:List[int] = self.results[f'{predictor}_subgoal_correct'].tolist()
        success_rate = correctnesses.count(1)/(correctnesses.count(1) + correctnesses.count(0))
        self.logger.info(f"Predictor: {predictor}, gt empty rate {correctnesses.count(-1)/len(correctnesses):.4}, success rate: {success_rate:.4}")
        stat = np.array(self.results[f'{predictor}_subgoal_statistics'].tolist()) # N*2*3
        acc = stat.sum(axis=0)
        self.logger.info(f"   Acc: subj {acc[1][0]/acc[0][0]:.3}, pred {acc[1][1]/acc[0][1]:.3}, obj {acc[1][2]/acc[0][2]:.3}")
        return success_rate, acc

if __name__ == "__main__":
    da = DataAnalyzer()
    da.load_gpt_data()
    # da.load_bart_data()
    # da.calc_accuracy()