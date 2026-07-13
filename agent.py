from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

load_dotenv()

# Load the vector store
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# The LLM
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

###################################
# TOOLS
###################################

@tool
def check_informed_consent(protocol: str) -> str:
    """Check whether the protocol describes an acceptable consent procedure."""
    protocol = protocol.lower()

    if "verbal consent" in protocol:
        return "FAIL: Protocol specifies verbal consent only."

    if "written consent" in protocol:
        return "PASS"

    return "WARNING: Consent procedure not clearly described."

@tool
def check_sae_reporting(protocol: str) -> str:
    """Check whether the protocol specifies an unsafe serious-adverse-event reporting timeline."""
    protocol = protocol.lower()

    if "30 days" in protocol:
        return "FAIL: SAE reporting appears too slow."

    return "PASS"

@tool
def check_eligibility(protocol: str) -> str:
    """Check whether the protocol includes both inclusion and exclusion criteria."""
    protocol = protocol.lower()

    if "inclusion" not in protocol:
        return "WARNING: Inclusion criteria missing."

    if "exclusion" not in protocol:
        return "WARNING: Exclusion criteria missing."

    return "PASS"


tools = [
    check_informed_consent,
    check_sae_reporting,
    check_eligibility,
]
llm_with_tools = llm.bind_tools(tools)

###################################
# GRAPH NODES
###################################

# Node 1: retrieve relevant chunks from ICH-GCP
def retrieve(state: MessagesState):
    last_message = state["messages"][-1].content
    docs = vectorstore.similarity_search_with_score(
    last_message,
    k=5
    )

    for doc, score in docs:
        print("=" * 60)
        print("Score:", score)
        print(doc.page_content[:500])

    context = "\n\n".join([doc.page_content for doc, score in docs])
    return {"messages": state["messages"] + [
        SystemMessage(content=f"Relevant ICH-GCP sections:\n\n{context}")
    ]}

# Node 2: let the model decide which LangChain tools to call
def tool_check(state: MessagesState):
    protocol = next(
        msg.content
        for msg in reversed(state["messages"])
        if isinstance(msg, HumanMessage)
    )

    response = llm_with_tools.invoke([
        SystemMessage(content="""You select compliance-checking tools for a clinical-trial
protocol. Call every applicable tool. Pass the complete protocol text as the `protocol`
argument to each selected tool. Do not provide a prose answer yet."""),
        HumanMessage(content=protocol),
    ])
    return {"messages": [response]}


# Node 3: generate compliance review using retrieved context
def review(state: MessagesState):

    context = ""
    human_message = ""
    system_messages = []
    tool_results = []

    for msg in state["messages"]:
        if isinstance(msg, SystemMessage):
            system_messages.append(msg.content)

        if isinstance(msg, HumanMessage):
            human_message = msg.content

        if isinstance(msg, ToolMessage):
            tool_results.append(f"{msg.name}: {msg.content}")

    context = "\n\n".join(system_messages)

    prompt = f"""You are a clinical trial protocol compliance reviewer.

Use ONLY the retrieved ICH-GCP text below when identifying compliance issues.

If the retrieved text does not contain enough information to support a conclusion,
explicitly state that additional sections of the guideline are needed.

Retrieved ICH-GCP sections:

{context}

Protocol excerpt:

{human_message}

Deterministic tool results:

{chr(10).join(tool_results) or "No tools were called."}

Structure your response as:
1. COMPLIANCE ISSUES FOUND
2. RELEVANT ICH-GCP REFERENCES
3. RECOMMENDATIONS
4. TOOLS USED AND RESULTS

Under "TOOLS USED AND RESULTS", list every tool that was run and its exact
deterministic result. If no tools were run, state that explicitly.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"messages": [response]}

def human_review(state: MessagesState):
    report = state["messages"][-1].content

    decision = interrupt(
        {
            "draft_report": report,
            "message": "Review the compliance report before continuing."
        }
    )

    if decision.get("approved", False):
        return state

    return {
        "messages": state["messages"] + [
            HumanMessage(
                content=f"Reviewer feedback: {decision['feedback']}"
            )
        ]
    }

# Build the graph
graph = StateGraph(MessagesState)
graph.add_node("retrieve", retrieve)
graph.add_node("tool_check", tool_check)
graph.add_node("tools", ToolNode(tools))
graph.add_node("review", review)
graph.add_node("human_review", human_review)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "tool_check")
graph.add_conditional_edges(
    "tool_check",
    tools_condition,
    {"tools": "tools", "__end__": "review"},
)
graph.add_edge("tools", "review")
graph.add_edge("review", "human_review")
graph.add_edge("human_review", END)
memory = InMemorySaver()
app = graph.compile(
    checkpointer=memory
)

# Visualize the graph
app.get_graph().draw_mermaid_png(output_file_path="graph.png")

# Test with a sample protocol excerpt
protocol_excerpt = """
The investigator will obtain verbal consent from participants before enrollment.
Adverse events will be collected at each visit but serious adverse events 
will be reported to the sponsor within 30 days.
"""

print("Running compliance review...\n")

config = {
    "configurable": {
        "thread_id": "trial_review_001"
    }
}

# Run until the graph pauses
result = app.invoke(
    {
        "messages": [
            HumanMessage(
                content=f"Review this protocol excerpt for ICH-GCP compliance:\n\n{protocol_excerpt}"
            )
        ]
    },
    config=config,
)

# Show the draft report
interrupt_data = result["__interrupt__"][0].value

print("\n" + "=" * 70)
print(interrupt_data["message"])
print("=" * 70)
print(interrupt_data["draft_report"])

# Simulate the human reviewer
decision = input("\nApprove report? (y/n): ")

if decision.lower() == "y":
    resume = app.invoke(
        Command(resume={"approved": True}),
        config=config,
    )
else:
    feedback = input("Enter reviewer comments: ")

    resume = app.invoke(
        Command(
            resume={
                "approved": False,
                "feedback": feedback,
            }
        ),
        config=config,
    )

print("\nGraph finished.")

print("\nFinal report:")
print("=" * 70)
print(resume["messages"][-1].content)
