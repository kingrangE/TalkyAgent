from state import TalkyState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import tools
load_dotenv()

llm = ChatOpenAI(model = "gpt-5-nano")
AUDIO_FILENAME = "recorded_audio.wav"

def chatbot(state: TalkyState):
    
    message = input("User : ")
    if message == "exit" :
        exit(0)
    state["messages"].append(HumanMessage(content=message))

    return {"messages": [llm.invoke(state["messages"])]}

def speech_to_text(state: TalkyState):
    audio_data = tools.record_audio()
    with open(AUDIO_FILENAME, "wb") as f:
        f.write(audio_data.get_wav_data())
    
def text_to_speech(state: TalkyState):
    tools.tts(state["messages"][-1].content)


def get_graph():
    graph = StateGraph(TalkyState)
    graph.add_node("chatbot",chatbot)
    graph.add_node("tts",text_to_speech)
    graph.set_entry_point("chatbot")
    graph.add_edge("chatbot","tts")
    graph.add_edge("tts","chatbot")

    return graph.compile()