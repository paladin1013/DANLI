# %%
import os
import sys
from typing import List, Dict, cast
import openai
import logging
openai.organization = "org-p5ug2Pool5bdCna5a285PeCU"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.proxy = {"http":"http://localhost:7890", "https":"https://localhost:7890"}

class GPTAPI:
    def __init__(self, provider="openai", model="gpt-3.5-turbo", log_level=logging.INFO):
        self.messages = []
        self.provider = provider
        self.model = model
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
            temperature=0.1)
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
    
if __name__ == "__main__":
    api = GPTAPI(log_level=logging.DEBUG)