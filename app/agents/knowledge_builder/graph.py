from langgraph.graph import StateGraph, END

from app.agents.knowledge_builder.state import KnowledgeBuilderState
from app.agents.knowledge_builder.nodes import extract_knowledge_graph_node
from app.agents.knowledge_builder.tools import write_knowledge_graph_to_neo4j


async def save_graph_to_database_node(state: KnowledgeBuilderState):
    """Save extracted graph elements to Neo4j database."""
    if state.errors:
        return {}
    try:
        await write_knowledge_graph_to_neo4j(state.project_id, state.extracted_graph)
    except Exception as e:
        return {"errors": state.errors + [f"Neo4j write error: {e}"]}
    return {}


# Define workflow
workflow = StateGraph(KnowledgeBuilderState)

# Add nodes
workflow.add_node("extract", extract_knowledge_graph_node)
workflow.add_node("save", save_graph_to_database_node)

# Set execution flow
workflow.set_entry_point("extract")
workflow.add_edge("extract", "save")
workflow.add_edge("save", END)

# Compile LangGraph
knowledge_builder_graph = workflow.compile()
