import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from agents.base_agent import AgentState
from agents.llm_agent import call_llm
from agents.retriever_agent import take_action
from agents.finalize_agent import finalize_answer
from tools.text_retriever_tool import text_retriever_tool
from tools.image_retriever_tool import image_retriever_tool, caption_and_index_images
from agents.loader_agent import load_and_split_text, extract_images_and_tables
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import system_prompt, TEXT_DB_PATH, IMAGE_DIR, PDF_PATH, TEXT_EMBEDDING_MODEL, LLM_MODEL_NAME

# === Load Environment Variables ===
print("🔧 Loading environment variables...")
load_dotenv()

# === Load Tools ===
print("🔧 Loading tools...")
tools = [text_retriever_tool]
tools_dict = {tool.name: tool for tool in tools}

# === Step 1: Preprocessing - Load + Split Text ===
print("\n📄 Step 1: Loading and splitting text chunks...")
chunks = load_and_split_text()
print(f"✅ Loaded and split {len(chunks)} chunks.")

# === Step 2: Initialize Embeddings and Vector Store ===
print("\n📚 Step 2: Initializing embeddings and Chroma vector store...")
embedding = HuggingFaceEmbeddings(model_name=TEXT_EMBEDDING_MODEL)

if not os.path.exists(TEXT_DB_PATH):
    print(f"📁 Creating vector store directory at: {TEXT_DB_PATH}")
    os.makedirs(TEXT_DB_PATH, exist_ok=True)
    text_vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=TEXT_DB_PATH)
    print("✅ Vector store created and populated.")
else:
    print(f"📂 Loading existing vector store from: {TEXT_DB_PATH}")
    text_vectorstore = Chroma(embedding_function=embedding, persist_directory=TEXT_DB_PATH)
    print("✅ Vector store loaded.")

# # === Step 3: Preprocessing - Extract Images and Caption ===
# print("\n🖼️ Step 3: Extracting and captioning images from PDFs...")

# # Uncomment these if needed
# extract_images_and_tables(PDF_PATH, IMAGE_DIR)
# # caption_and_index_images()

# print("✅ Image processing complete (if enabled).")

# === Step 4: Build Agentic RAG Graph ===
MAX_TOOL_CALLS = 3

def should_continue(state: AgentState):
    # Prevent infinite loop
    # if state.get("tool_calls_made", 0) >= MAX_TOOL_CALLS:
    #     print("⚠️ Max tool call loops reached. Finalizing...")
    #     return False
    
    # Otherwise follow tool_calls existence
    last_msg = state["messages"][-1]
    return hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0

# def should_continue(state: AgentState):
#     """Check if the last message contains tool calls."""
#     result = state['messages'][-1]
#     return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

print("\n🔗 Step 4: Building agentic graph...")
llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0).bind_tools(tools)

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm(llm, tools_dict, system_prompt))
graph.add_node("retriever", take_action)
graph.add_node("finalizer", finalize_answer)

graph.add_conditional_edges("llm", should_continue, {
    True: "retriever",
    False: "finalizer"  # Finalize only when LLM stops calling tools
})
graph.add_edge("retriever", "llm")  # Let LLM summarize ToolMessage
graph.add_edge("finalizer", END)
graph.set_entry_point("llm")

rag_agent = graph.compile()
print("✅ Agentic graph compiled successfully!")

# === Save Graph Structure ===
print(rag_agent.get_graph().draw_ascii())

# === Agentic Interaction Loop ===
print("\n🚀 Agentic Multimodal RAG Ready!")
print("💬 Type your question below or 'exit' to quit.")
while True:
    query = input("\n🧠 Your question: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting agent...")
        break

    print("🤖 Thinking...")
    result = rag_agent.invoke({"messages": [HumanMessage(content=query)]})

    print("\n📢 [ANSWER]")
    print(result["messages"][-1].content)
