from llm import LLM
from typing import Any,Optional
from prompts import NEExtractionTemplate,Text2CypherTemplate
import speech_recognition as sr
from playsound import playsound
from gtts import gTTS as gt
import neo4j
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.types import Neo4jGraph

class VoiceTool :
    def __init__(self):
        r = sr.Recognizer()
        mic = sr.Microphone()
    def tts(self, content: str):
        """전달받은 text를 tts를 거쳐 음성파일을 저장하고 재생"""
        speech = gt(content,lang="en")
        speech.save("ai_response.wav")
        self.__play_audio("ai_response.wav")

    def stt(self):
        """녹음하고, stt를 거쳐 text를 반환"""
        with sr.Microphone() as source: 
            print("말해보세요!")
            audio=r.listen(source,100) 
        try:
            transcript=r.recognize_google(audio, language="en") 
            return transcript
        except sr.UnknownValueError: 
            return "다시 말해주세요."
        except sr.RequestError as e:
            return  "작동되지 않았습니다. 오류코드: {0}".format(e)

    def __play_audio(file_name:str):
        playsound(file_name)

class Neo4jGraphTool :
    def __init__(self):
        self.extraction_template = NEExtractionTemplate()
        self.cyper_template = Text2CypherTemplate()
        self.llm = LLM()
    def extract_entity(
            self, 
            schema:dict[str,Any], 
            examples:str,
            text:str, 
            history:str
        )-> str:
        prompt = self.extraction_template.format(schema=schema,
                             examples=examples,
                             text=text,
                             history=history)
        
        # prompt가 길어질 수록 많은 노드와 관계가 추출될 수 있으므로 prompt 길이에 따라 max_new_token을 결정하도록 함.
        if len(prompt)/5 < 512:
            max_new_token = 512
        else : 
            max_new_token = 512

        result = self.llm.invoke(prompt,max_new_token=max_new_token) 
        result = result[result.find('{'):result.rfind('}')+1]
        return result
    
    def get_cyper_query(self,
                        schema: Optional[str] = None,
                        examples: Optional[str] = None,
                        query:str = ""):
        
        prompt = self.cyper_template.format(schema=schema,examples=examples,query_text=query)
        result = self.llm.invoke(prompt,max_new_token=512)
        return result
    
    async def write(self,graph_json:dict):
        with neo4j.GraphDatabase.driver("bolt://13.219.100.55",auth=("neo4j","potatos-voltages-harpoon")) as driver:
                writer = Neo4jWriter(driver)
                graph = Neo4jGraph(
                    nodes = graph_json["nodes"],
                    relationships = graph_json["relationships"],
                )
                await writer.run(graph)
                print("저장 완료")
    
if __name__ == "__main__":
    r=sr.Recognizer() 