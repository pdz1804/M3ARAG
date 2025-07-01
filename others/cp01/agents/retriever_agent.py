from agents.base_agent import AgentState
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from tools import text_retriever_tool, image_retriever_tool

tools = [text_retriever_tool, image_retriever_tool]  # List of tools to be used by the agent

tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
            
        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {
        "messages": results,
        "tool_calls_made": state.get("tool_calls_made", 0) + 1
    }

# def take_action(state: AgentState) -> AgentState:
#     """Execute tool calls from the LLM's response."""

#     tool_calls = state['messages'][-1].tool_calls

#     # Optionally guard against redundant tool execution
#     prev_tool_ids = {
#         m.tool_call_id for m in state['messages'] if isinstance(m, ToolMessage)
#     }

#     results = []
#     for t in tool_calls:
#         if t['id'] in prev_tool_ids:
#             print(f"Tool {t['name']} with ID {t['id']} already executed. Skipping.")
#             continue

#         print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

#         if t['name'] not in tools_dict:
#             print(f"\nTool: {t['name']} does not exist.")
#             result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
#         else:
#             result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
#             print(f"Result length: {len(str(result))}")

#         results.append(
#             ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result))
#         )

#     print("Tools Execution Complete. Back to the model!")
    
#     # ✅ Append tool results to previous message history
#     return {"messages": state["messages"] + results}

