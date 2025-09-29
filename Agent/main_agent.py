"""
Main Agent가 하는 일

1. 사용자의 발화 텍스트를 받아 의도를 파악
    - LoRA Adapter를 학습시켜서 의도파악을 하도록 해야할 것 같음.
    - 1.2B의 지식으로는 쉽지 않은듯
2. 사용자의 의도에 따라 참조해야 할 Memory 추출
    - 정보 retriever이 필요하다면, (메모리,필요한 정보) 전달
        - 응답 (False, 추출 정보)
    - 정보 update가 필요하다면, (메모리, 업데이트 할 정보) 전달
        - 응답 (True, None)
3. 얻은 결과를 토대로 답변 생성
4. 답변 스스로 평가
    - 적절 시 응답 : (False, None)
    - 부적절 시 응답 : (True, 부적절한 이유)
"""

import json
from outlines import Generator
import outlines
import os
import torch
from huggingface_hub import snapshot_download
from langchain_community.llms import Outlines
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List,Tuple,Optional, Literal
TALK_PROMPT="""
You are a conversation expert who leads the conversation with the other person flexibly.
Please lead the conversation in a fun way by responding flexibly and asking questions according to the other person's words.

Try to keep the other person interested in the conversation to lead the conversation.
"""
ANALYSIS_PROMPT ="""
당신은 대화 기록을 통해 user 마지막 말의 의도를 파악하는 전문 심리학자입니다. 
대화 기록의 주제는 영어 assistant와 user 대화이며, user는 영어를 잘하지 못합니다.
user의 모든 말은 하단의 4가지 의도를 가지며, 대화 기록을 통해 user의 의도를 파악하여 의도 번호만 말씀해주시면 됩니다.

1. 대화 이어가기 요청 
    - 설명
        - user가 질문에 답하거나, 질문을 하는 등 assistant와 대화를 이어가길 원하는 의도
    - 조건
        - user의 마지막 말이 영어 문장
2. 영어 번역 요청 
    - 설명
        - user가 영어로 표현하는 방법을 몰라 한국어로 말했을 때, 영어로 표현해주기를 바라는 의도
    - 조건
        - user의 마지막 말이 한국어
        - 대화 맥락과 맞는 말이나 한국어로 말하였을 때
3. 표현 개선 요청
    - 설명
        - user가 이전에 했던 표현을 더 낫게 개선하기를 원하는 의도를 가짐.
    - 조건
        - user가 이전에 나눈 영어 대화가 존재
4. 평가
    - 설명
        - user가 이전에 했던 대화 내용 등을 평가하는 의도를 가짐.
    - 조건
        - user와 이전에 나눈 영어 대화가 존재
        - 이전에 나눈 대화가 좋았다, 나빴다 등의 평가 내용이 존재함.
"""
class EvaluationResult(BaseModel):
    status : Literal["success","fail"] = Field(
        ...,
        description="평가 결과. 'success'는 평가 적합, 'fail'은 평가 부적합"
    )
    reason : str = Field(
        ...,
        description="평가 결과에 대한 이유를 설명합니다."
    )

class MainAgent:
    def __init__(self):
        # exaone 없으면 download
        if "exaone-4" not in os.listdir():
            snapshot_download(repo_id="LGAI-EXAONE/EXAONE-4.0-1.2B",local_dir="exaone-4")
        self.model = AutoModelForCausalLM.from_pretrained("exaone-4",dtype=torch.float16,device_map="auto",)
        self.tokenizer = AutoTokenizer.from_pretrained("exaone-4")
        self.evaluate_model = outlines.from_transformers(
            self.model,
            self.tokenizer
        )
        self.model.eval()
        self.messages = [{"role":"system","content":TALK_PROMPT}]

    def execute(self):
        """전체 로직, 입력 받고 생각->행동 반복"""
        while True:
            message = input("user :")
            if message == "quit" or message == "q":
                exit(0)
            # intention = self.think(message)
            intention = int(input())
            self.act(intention,message)
            
    def think(self,message:str) -> int:
        """의도 파악을 위한 생각 단계, 이후에 여기는 Adapter를 이용한 예측 또는 Model을 이용한 분류로 대체 예정"""
        intention = self.analysis_intention(message = message) # 의도 분석
        return intention
    
    def act(self, intention: int, message: str) :
        """의도에 따른 행동 분류"""
        if intention == 1 :
            # 현재 메시지를 전달하여 답변 얻기
            self.talk(message=message)

        elif intention == 2 :
            # 현재 보낸 메시지를 번역
            print(self.translate(message=message))

        elif intention == 3 :
            # 이전에 보낸 user의 답변 전송
            print(self.improve_expression(message=self.messages[-2]) )
            """
            개선 : 
            talk에서 Improve_expression이 내장되면 좋을듯?
            """
        else :
            # 현재 메시지가 피드백이므로 현재 메시지를 전달
            print(self.save_feedback(message=message))

    def talk(self,message: str)-> None:
        """User와 대화를 진행하는 함수"""
        self.messages.append({"role":"user","content":message}) # 사용자 입력 메시지 추가
        # chat template 변환
        input_ids = self.tokenizer.apply_chat_template(self.messages,add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        # 생성
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens = len(input_ids)+512,
                do_sample = False
            )

        # 결과 decoding & extract output
        result = self.tokenizer.decode(output[0],skip_special_tokens=True)
        result = result.split('</think>')[-1].strip()

        # 대화 내역에 추가
        self.messages.append({"role":"assistant","content":result})
        print("AI :",result)

    def translate(self,message: str) -> str :
        """한국어로 말한 문장을 영어로 번역해주는 단계"""
        TRANSLATION_EVALUATION_PROMPT="지금까지의 했던 대화를 토대로 보았을 때, 다음 문장이 자연스러운 답변인지 알려주세요."
        prompt = [{"role":"system","content":f"당신은 영어 통번역 전문가입니다. 다음 대사를 영어권 사람들이 일상적으로 구사하는 표현을 사용하여 영어로 번역해주세요. \n대사 : {message}"}]
        result = self.invoke(prompt)
        success,reason = self.evaluate(prompt=TRANSLATION_EVALUATION_PROMPT,sentence_to_evaluation=result)

        if not success :
            prompt.append({"role":"assistant","content":result})
            prompt.append({"role":"user","content":f"당신의 번역은 다음의 이유로 부적절합니다. 이유를 고려하여 개선한 번역본을 전달해주세요.\n이유: {reason}"})
            result = self.invoke(prompt,thinking=True)
            success,reason = self.evaluate(prompt=TRANSLATION_EVALUATION_PROMPT,sentence_to_evaluation=result)

        return result
    
    def improve_expression(self, message: str) -> str :
        """표현 개선 함수"""
        prompt = [{"role":"system","content":f"당신은 영어 회화 전문가입니다. 다음 영어 표현을 영어권 사람들이 일상적으로 사용하는 표현으로 개선해주세요. 또한 표현을 더 풍부하게 개선해주세요. \n대사 : {message}"}]
        result = self.invoke(prompt)
        status,reason = self.evaluate(sentence_to_evaluation=result)

        if status == "success" :
            prompt.append({"role":"assistant","content":result})
            prompt.append({"role":"user","content":f"당신의 개선본은 다음의 이유로 부적절합니다. 이유를 고려하여 다시 개선해주세요.\n이유: {reason}"})
            result = self.invoke(prompt)
            status,reason = self.evaluate(prompt="이전의 나눈 대화를 토대로 다음의 문장이 개선본으로 적절한지 알려주세요.",sentence_to_evaluation=result)
        
        return result

    def save_feedback(self, message: str) -> str:
        """사용자가 피드백한 내용을 저장하기 위한 함수"""
        system_prompt = [{"role": "system", "content": "당신은 사용자의 피드백을 통해 사용자가 현재 대화의 어떤 부분을 어떻게 생각하는지 요약하는 일의 전문가입니다. 다음 제공되는 user와 assistant의 대화 그리고 사용자의 피드백을 토대로 사용자의 생각을 요약해주세요."}]
        full_prompt = system_prompt + self.messages + [{"role": "user", "content": message}]
        
        feedback = self.invoke(full_prompt)
        return feedback

    
    def evaluate(self, prompt: str, sentence_to_evaluation: str) -> Tuple[bool,Optional[str]]:
        """목적에 맞게 잘 행동했는지 평가하기 위한 evaluate 함수"""
        generator = outlines.Generator(self.evaluate_model,EvaluationResult)
        input = self.messages + [{"role":"system","content":f"{prompt}\n{sentence_to_evaluation}"}]
        input = self.chat_list_to_str(input)
        result = json.loads(generator(input, max_new_tokens=128))
        return result["status"],result["reason"]
    
    def chat_list_to_str(self,chat_list: List[dict]) -> str:
        """evaluator에 넣기 위해 List 자료형을 str 자료형으로 변환해주는 함수"""
        result = ""
        for chat in chat_list :
            if chat["role"] == "user":
                result += "\nUser :"+chat["content"]
            elif chat["role"] == "assistant":
                result += "\nAI :"+chat["content"]
        return result
            

    def invoke(self,messages: str, thinking: bool = False) -> str:
        """추론 단계"""
        input_ids = self.tokenizer.apply_chat_template(messages,
                                                       add_generation_prompt=True, 
                                                       return_tensors="pt",
                                                       enable_thinking = thinking).to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens = len(input_ids)+512,
                do_sample = False,
            )

        result = self.tokenizer.decode(output[0],skip_special_tokens=True)
        result = result.split('</think>')[-1].strip()

        return result
    
if __name__=="__main__":
    agent = MainAgent() 
    agent.execute()
