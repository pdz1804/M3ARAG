# orchestrator_runner.py

from agents.text_agent import TextRAGAgent
from agents.generalize_agent import GeneralizeAgent
from agents.finalize_agent import FinalizeAgent
from agents.multi_agent_runner import MultiAgentRunner

def answer_question(question: str, use_text=True, use_image=False) -> str:
    runner = MultiAgentRunner()

    if use_text:
        runner.register_agent(TextRAGAgent())

    # TODO: Future image agent could go here...
    if use_image:
        pass

    runner.register_agent(GeneralizeAgent())
    runner.register_agent(FinalizeAgent())

    final_output = runner.run({"question": question})

    print("\n✅ Final Answer:")
    return final_output


