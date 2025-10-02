import os
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch
class LLM :
    def __init__(self,model_name:str="exaone-4"):
        if model_name not in os.listdir():
            snapshot_download(repo_id=model_name,local_dir=model_name.split('/')[-1])
        self.model = AutoModelForCausalLM.from_pretrained(model_name,dtype=torch.float16,device_map="auto",)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def invoke(self,messages: str, thinking: bool = False, max_new_token:int = 512) -> str:
        """추론 단계"""
        input_ids = self.tokenizer.apply_chat_template(messages,
                                                       add_generation_prompt=True, 
                                                       return_tensors="pt",
                                                       enable_thinking = thinking).to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens = len(input_ids)+max_new_token,
                do_sample = False,
            )

        result = self.tokenizer.decode(output[0],skip_special_tokens=True)
        result = result.split('</think>')[-1].strip()

        return result