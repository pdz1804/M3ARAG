from langchain_core.messages import SystemMessage
from langgraph.graph import END

# def call_llm(llm, tools_dict, system_prompt):
#     def node(state):
#         messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
#         response = llm.invoke(messages)
#         return {"messages": [response]}
    
#     return node

def call_llm(llm, tools_dict, system_prompt):
    def node(state):
        state.setdefault("depth", 0)
        if state["depth"] > 25:
            raise RuntimeError("Recursion depth exceeded")
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        response = llm.invoke(messages)
        
        # 🔍 DEBUG: Print tool call plan if any
        if hasattr(response, "tool_calls"):
            print("\n[LLM TOOL CALL PLANS]")
            for tool_call in response.tool_calls:
                print(f"Tool: {tool_call['name']}")
                print(f"Args: {tool_call['args']}")
                print("-" * 40)
        else:
            print("\n[LLM OUTPUT IS NOT A TOOL CALL]")
            print(f"Response Content:\n{response.content}")
        
        return {
            "messages": state["messages"] + [response],
            "depth": state["depth"] + 1
        }
    return node



