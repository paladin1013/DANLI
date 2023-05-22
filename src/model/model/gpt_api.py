import os
import sys
from typing import Optional, List, Dict, cast
import openai
import logging
import jsonlines
from tqdm import tqdm
from pydantic import BaseModel
openai.organization = "org-p5ug2Pool5bdCna5a285PeCU"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.proxy = {"http":"http://localhost:7890", "https":"https://localhost:7890"}


class GPT4Session(BaseModel):
    id:str
    prompts:List[str]
    replies:List[str]
    error:Optional[str]

class GPTAPI:
    def __init__(self, provider="openai", model="gpt-3.5-turbo", log_level=logging.INFO, temperature=0):
        self.messages = []
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.logger = logging.getLogger("gpt_api")
        self.logger.setLevel(log_level)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        
    def check_connectivity(self):
        self.logger.debug(f"Checking gpt api connectivity")
        chat_messages = []
        chat_messages.append(
            {
                'role':'user',
                'content':"Please reply yes if everything is ok."
            })
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages = chat_messages,
            temperature=self.temperature)
        completion = cast(Dict[str, List[Dict[str, Dict[str, str]]]], completion)
        reply:str = completion['choices'][0]['message']['content']
        self.logger.debug(f"Received reply:\n{reply}\n")
        if reply:
            return True
        return False
    def send(self, messages:List[str]):
        replies:List[str] = []
        chat_messages = []
        for message in messages:
            self.logger.debug(f"Sending message: {message}")
            chat_messages.append(
                {
                    'role':'user',
                    'content':message
                })
            if self.provider == "openai":
                completion = openai.ChatCompletion.create(
                    model=self.model,
                    messages = chat_messages,
                    temperature=0.1)
                completion = cast(Dict[str, List[Dict[str, Dict[str, str]]]], completion)
                reply:str = completion['choices'][0]['message']['content']
            else:
                raise NotImplementedError(f"Provider {self.provider} is not implemented yet.")
            self.logger.debug(f"Received reply:\n{reply}\n")
            replies.append(reply)
            chat_messages.append(
                {
                    'role':'assistant',
                    'content':reply
                })
        return replies
    
    def generate_jsonl(self, jsonl_file_path:str, sessions:Dict[str, GPT4Session]):
        """For GPT4 queries with batch message"""

        jsonl_data = []
        for id, session in tqdm(sessions.items()):
            sample_data = {}
            sample_data['id'] = id
            sample_data['text'] = session.prompts
            sample_data['first_text_length'] = len(session.prompts[0])
            sample_data['all_answers'] = []
            jsonl_data.append(sample_data)
        with jsonlines.open(jsonl_file_path, "w") as writer:
            writer.write_all(jsonl_data)
        self.logger.info(f"Generated jsonl file at {jsonl_file_path}")
    
    def load_response_jsonl(self, jsonl_file_path:str):
        """For GPT4 queries with batch message"""
        sessions:Dict[str, GPT4Session] = {}
        with jsonlines.open(jsonl_file_path) as reader:
            for obj in reader:
                sessions[obj['id']] = GPT4Session(id=obj['id'], prompts=obj['text'], replies=obj['all_answers'], error=obj['error'] if 'error' in obj.keys() else None)
        return sessions
    
if __name__ == "__main__":
    api = GPTAPI(log_level=logging.DEBUG)