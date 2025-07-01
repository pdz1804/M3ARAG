import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from config import system_prompt, TEXT_DB_PATH, IMAGE_DIR, PDF_PATH, TEXT_EMBEDDING_MODEL, LLM_MODEL_NAME
from tools.text_retriever_tool import text_retriever_tool
from tools.image_retriever_tool import image_retriever_tool, caption_and_index_images
from agents.loader_agent import load_and_split_text, extract_images_and_tables
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain.memory import ConversationBufferMemory

# === Step 1: Load Environment ===
print("🔧 Loading environment variables...")
load_dotenv()

# === Step 2: Preprocessing - Text ===
print("\n📄 Step 1: Loading and splitting text chunks...")
chunks = load_and_split_text()
print(f"✅ Loaded and split {len(chunks)} chunks.")

# === Step 3: Initialize Embeddings and Chroma ===
print("\n📚 Step 2: Initializing embeddings and Chroma vector store...")
embedding = HuggingFaceEmbeddings(model_name=TEXT_EMBEDDING_MODEL)

if not os.path.exists(TEXT_DB_PATH):
    os.makedirs(TEXT_DB_PATH, exist_ok=True)
    text_vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=TEXT_DB_PATH)
    print("✅ Vector store created and populated.")
else:
    text_vectorstore = Chroma(embedding_function=embedding, persist_directory=TEXT_DB_PATH)
    print("✅ Loaded existing vector store.")

# === Step 4: Optional Image Processing ===
# extract_images_and_tables(PDF_PATH, IMAGE_DIR)
# caption_and_index_images()

# === Step 5: Create Agent with Tools ===
print("\n🤖 Initializing agent...")

# ✅ Define agent creator function
def create_agent(llm, tools):
    MEMORY_KEY = "chat_history"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create the agent
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

    # Wrap with executor
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,   # ✅ add memory
        verbose=True,
    )
    
    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=None)
    return agent_executor

tools = [text_retriever_tool]  # You can add image_retriever_tool if needed
llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0).bind_tools(tools)
agent_executor = create_agent(llm, tools)

# === Step 6: Agent Loop ===
print("\n🚀 Agentic Multimodal RAG Ready!")
print("💬 Type your question below or 'exit' to quit.")

while True:
    query = input("\n🧠 Your question: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting agent...")
        break

    print("🤖 Thinking...")
    result = agent_executor.invoke({"input": query})
    print("\n📢 [ANSWER]")
    print(result["output"])
