from langgraph.graph import StateGraph, END

from app.agents.knowledge_builder.state import KnowledgeBuilderState
from app.agents.knowledge_builder.nodes import (
    preprocess_chunk_node,
    generate_embeddings_node,
    extract_graph_node,
    store_vectors_node,
    store_graph_node,
    save_results_node,
)

# Define workflow
workflow = StateGraph(KnowledgeBuilderState)

# Add all agent workflow nodes
workflow.add_node("preprocess", preprocess_chunk_node)
workflow.add_node("embed", generate_embeddings_node)
workflow.add_node("extract", extract_graph_node)
workflow.add_node("store_vectors", store_vectors_node)
workflow.add_node("store_graph", store_graph_node)
workflow.add_node("save_results", save_results_node)

# Set entry point
workflow.set_entry_point("preprocess")

# Connect nodes sequentially
workflow.add_edge("preprocess", "embed")
workflow.add_edge("embed", "extract")
workflow.add_edge("extract", "store_vectors")
workflow.add_edge("store_vectors", "store_graph")
workflow.add_edge("store_graph", "save_results")
workflow.add_edge("save_results", END)

# Compile the agent graph
knowledge_builder_graph = workflow.compile()
