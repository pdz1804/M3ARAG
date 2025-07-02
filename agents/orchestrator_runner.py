# agents/orchestrator_runner.py
from agents.multi_agent_runner import MultiAgentRunner
from agents.registry import AGENTS

def answer_question(question: str, use_text=True, use_image=False) -> str:
    runner = MultiAgentRunner()

    if use_text:
        runner.register_agent(AGENTS["TextRAGAgent"])

    if use_image:
        runner.register_agent(AGENTS["ImageRAGAgent"])

    runner.register_agent(AGENTS["GeneralizeAgent"])
    runner.register_agent(AGENTS["FinalizeAgent"])

    return runner.run({"question": question})


