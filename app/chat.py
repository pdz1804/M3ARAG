import logging

from rag_text.loader import load_documents_from_folder
from rag_text.vectorstore import get_retriever
from rag_text.chain import build_rag_chain

# === Agent import (for text-agent / generalize / finalize)
from agents.orchestrator_runner import answer_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_rag_chat(use_agent: bool = False):
    """Step 2: Load, embed, and run interactive RAG chat on extracted PDFs or via agents."""
    try:
        if use_agent:
            print("\n🤖 Agent Mode Enabled: TextAgent → GeneralizeAgent → FinalizeAgent")
            print("Type 'exit' or press Ctrl+C to quit.\n")

            while True:
                try:
                    question = input("❓ Ask: ").strip()
                    if question.lower() in {"exit", "quit"}:
                        print("👋 Goodbye!")
                        break
                    answer = answer_question(
                        question, use_text=True, use_image=False
                    )
                    print(f"\n💬 Final Answer:\n{answer}\n" + "-" * 80)
                except KeyboardInterrupt:
                    print("\n👋 Exiting...")
                    break
                except Exception as err:
                    print(f"❌ Agent Error: {err}")

        else:
            logger.info("📚 Loading documents from extract/pdf and store...")
            docs = load_documents_from_folder()

            if not docs:
                logger.warning("⚠️ No documents found. Did you forget to extract them?")
                return

            retriever = get_retriever(docs, persist_path="vectorstores/text_db")
            chain = build_rag_chain(retriever)

            print("\n✅ RAG is ready. Ask questions below.")
            print("Type 'exit' or press Ctrl+C to quit.\n")

            while True:
                try:
                    question = input("❓ Ask: ").strip()
                    if question.lower() in {"exit", "quit"}:
                        print("👋 Goodbye!")
                        break
                    answer = chain.invoke(question)
                    print(f"\n💬 {answer}\n" + "-" * 80)
                except KeyboardInterrupt:
                    print("\n👋 Exiting...")
                    break
                except Exception as err:
                    print(f"❌ Error: {err}")

    except Exception as e:
        logger.error(f"❌ Failed to start RAG chat: {e}")
        
        
        