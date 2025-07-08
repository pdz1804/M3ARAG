# pipeline/run_chat.py
def run_chat(pipeline):
    """Run the chat loop after pipeline is ready."""
    print("\n🤖 M3A Chat Ready! Ask me anything about the documents.")
    print("🔁 Type 'exit' or Ctrl+C to quit.\n")

    try:
        while True:
            question = input("❓ Ask: ").strip()
            if question.lower() in {"exit", "quit"}:
                print("👋 Exiting chat.")
                break
            pipeline.process_query(question)
    except KeyboardInterrupt:
        print("\n👋 Chat interrupted.")
