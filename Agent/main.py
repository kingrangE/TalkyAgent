from graph import get_graph
from langchain_openai import ChatOpenAI
if __name__ == "__main__" :
    graph = get_graph()
    graph.invoke({
        "messages" : [],
        "llm" : ChatOpenAI(model="gpt-5-nano")
    })