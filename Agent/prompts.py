from typing import Any,Optional,List,Literal
from exceptions import PromptMissingPlaceholderError,PromptMissingInputError

class PromptTemplate :
    DEFAULT_SYSTEM_INSTRUCTIONS: str = ""
    DEFAULT_TEMPLATE: str =""
    EXPECTED_INPUTS: List[str] = []
    def __init__(
            self, 
            template: Optional[str] = None, 
            expected_inputs:Optional[List[str]] = None,
            default_system_instructions:Optional[str] = None
    )->None:
        self.template = template or self.DEFAULT_TEMPLATE
        self.expected_inputs = expected_inputs or self.EXPECTED_INPUTS
        self.default_system_instructions = (
            default_system_instructions or self.DEFAULT_SYSTEM_INSTRUCTIONS
        )

        for e in self.expected_inputs:
            if f"{{{e}}}" not in self.template:
                raise PromptMissingPlaceholderError(
                    f"template에 다음 placeholder가 존재하지 않습니다. {e}"
                )
    def __format(self, **kwargs: Any) -> str:
        for e in self.EXPECTED_INPUTS:
            if e not in kwargs:
                raise PromptMissingInputError(f"다음 파라미터는 필수입니다. '{e}'")
        return self.template.format(**kwargs)
    
    def format(self,*args:Any ,**kwargs: Any) -> str:
        data = kwargs # keyword arguments 등록
        data.update({k: v for k, v in zip(self.expected_inputs, args)}) # positional argument가 있는 경우 추가
        return self.__format(**data)
        
class NEExtractionPrompt(PromptTemplate):
    def __init__(self,lang:Literal['en','kr']='kr'):
        if lang=='kr':
            # 한국어 prompt 반환
            DEFAULT_TEMPLATE = """
        당신은 지식 그래프(Knowledge Graph)를 구축하기 위해 구조화된 형식으로 정보를 추출하도록 설계된 최고 수준의 알고리즘입니다.

        다음 텍스트에서 엔티티(노드)를 추출하고 해당 유형을 명시하세요.
        또한, 이 노드들 간의 관계도 추출하세요.

        결과는 다음 형식을 사용하여 JSON으로 반환하세요:
        {{"nodes": [ {{"id": "0", "label": "사람", "properties": {{"이름": "홍길동"}} }}], "relationships": [{{"type": "알고있음", "start_node_id": "0", "end_node_id": "1", "properties": {{"언제부터": "2024-08-01"}} }}] }}

        (제공된 경우) 아래에 명시된 노드 및 관계 유형만 사용하세요:
        {schema}

        각 노드에 고유한 ID(문자열)를 할당하고, 이 ID를 사용하여 관계를 정의하세요.
        관계의 시작(source) 및 대상(target) 노드 유형과 관계의 방향을 반드시 준수해야 합니다.

        유효한 JSON 객체를 생성하려면 다음 규칙을 반드시 지켜주세요:

        JSON 외에 다른 추가 정보는 절대 반환하지 마세요.

        JSON을 감싸는 백틱(`)을 생략하고 JSON 자체만 출력하세요.

        JSON 객체는 리스트(list)로 감싸지 않은, 그 자체로 하나의 JSON 객체여야 합니다.

        속성 이름(Property names)은 반드시 큰따옴표("")로 묶어야 합니다.

        예시:
        {examples}

        입력 텍스트:
        {text}
        """
        else :
            # 영어 Prompt 반환
            DEFAULT_TEMPLATE = """
        You are a top-tier algorithm designed for extracting
        information in structured formats to build a knowledge graph.

        Extract the entities (nodes) and specify their type from the following text.
        Also extract the relationships between these nodes.

        Return result as JSON using the following format:
        {{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],
        "relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{"since": "2024-08-01"}} }}] }}

        Use only the following node and relationship types (if provided):
        {schema}

        Assign a unique ID (string) to each node, and reuse it to define relationships.
        Do respect the source and target node types for relationship and
        the relationship direction.

        Make sure you adhere to the following rules to produce valid JSON objects:
        - Do not return any additional information other than the JSON in it.
        - Omit any backticks around the JSON - simply output the JSON on its own.
        - The JSON object must not wrapped into a list - it is its own JSON object.
        - Property names must be enclosed in double quotes

        Examples:
        {examples}

        Input text:

        {text}
        """
        EXPECTED_INPUTS = ["text"]
        super().__init__(template=DEFAULT_TEMPLATE,expected_inputs=EXPECTED_INPUTS)


    def format(
        self,
        schema: dict[str, Any],
        examples: str,
        text: str = "",
    ) -> str:
        return super().format(text=text, schema=schema, examples=examples)

if __name__ == "__main__":
    kr = NEExtractionPrompt("kr").format(schema='',examples='',text="hi")
    en = NEExtractionPrompt("en").format(schema='',examples='',text="한국")