import os
import tiktoken
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

# --- NEW AGENT IMPORTS ---
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

# --- GLOBAL CONFIG & MODELS ---
CHROMA_PATH = "chroma"
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
# Notice: temperature=0 is critical for Agents. They need logic, not creativity.
model = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant", temperature=0)
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Setup Hybrid Retriever
data = db.get()
doc_objects = [Document(page_content=c, metadata=m) for c, m in zip(data['documents'], data['metadatas'])]
keyword_retriever = BM25Retriever.from_documents(doc_objects)
keyword_retriever.k = 5
vector_retriever = db.as_retriever(search_kwargs={"k": 5})
ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5])


# --- L1: THE AGENT'S TOOLBOX ---

@tool
def wonderland_search(query: str) -> str:
    """Use this tool FIRST whenever you need to find facts, characters, or events from the Alice in Wonderland book."""
    initial_results = ensemble_retriever.invoke(query)
    pairs = [[query, doc.page_content] for doc in initial_results]
    scores = reranker_model.predict(pairs)
    reranked_docs = sorted(zip(scores, initial_results), key=lambda x: x[0], reverse=True)
    final_docs = [doc for score, doc in reranked_docs[:3]]
    return "\n\n---\n\n".join([doc.page_content for doc in final_docs])

@tool
def calculator(expression: str) -> str:
    """
    Use this tool to solve math problems. 
    CRITICAL: The input MUST be a pure mathematical equation using ONLY numbers and operators (+, -, *, /).
    Do NOT pass variables, words, or letters. 
    Example of GOOD input: '3 * 452'
    Example of BAD input: 'result * 452'
    """
    try:
        # Strip out any random characters the AI might try to sneak in
        clean_expr = "".join(c for c in expression if c in "0123456789+-*/(). ")
        return str(eval(clean_expr, {"__builtins__": None}, {}))
    except Exception as e:
        return f"Math Error: You must use numbers only. Details: {e}"

# Register the tools
tools = [wonderland_search, calculator]


# --- L3: MEMORY GOVERNANCE ---
def count_tokens(text: str) -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4 

def compress_memory(chat_history: list, llm_model) -> list:
    print("\n⚙️ [SYSTEM]: Token limit reached. Compressing memory buffer...")
    history_text = "\n".join([f"{type(msg).__name__}: {msg.content}" for msg in chat_history])
    summary_prompt = f"Summarize this conversation concisely. Preserve facts and intent.\nLOG:\n{history_text}"
    summary_response = llm_model.invoke(summary_prompt)
    return [SystemMessage(content=f"Summary of previous conversation: {summary_response.content}")]


# --- AGENTIC ORCHESTRATION ---

SYSTEM_PROMPT = """
You are an intelligent Agent based on Alice in Wonderland. 
You are equipped with tools. You MUST use the tools to answer questions.
Do not guess. If a user asks about the book, use `wonderland_search`. 
If a user asks a math question, use `calculator`.
"""

# The Agent requires a specific prompt structure, including the "agent_scratchpad" 
# where it writes down its intermediate thoughts.
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 1. Bind the tools to the LLM
agent = create_tool_calling_agent(model, tools, prompt)

# 2. Create the Executor (We set verbose=True so you can see the AI "thinking")
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def query_agent(query_text: str, chat_history: list = None):
    if chat_history is None:
        chat_history = []
    
    # We use the executor instead of the direct chain
    response = agent_executor.invoke({
        "input": query_text,
        "chat_history": chat_history
    })
    
    return response["output"]

def main():
    chat_history = []
    MAX_HISTORY_TOKENS = 1000 
    
    print("🎩 Wonderland AGENT Initialized! (ReAct / Tools: ACTIVE)")
    
    while True:
        query_text = input("\n👤 You: ")
        if query_text.lower() in ["exit", "quit"]: break
        
        history_string = " ".join([msg.content for msg in chat_history])
        if count_tokens(history_string) > MAX_HISTORY_TOKENS:
            chat_history = compress_memory(chat_history, model)
        
        # Call the Agent
        answer = query_agent(query_text, chat_history)
        
        chat_history.append(HumanMessage(content=query_text))
        chat_history.append(AIMessage(content=answer))

        print(f"\n🤖 AI: {answer}")

if __name__ == "__main__":
    main()