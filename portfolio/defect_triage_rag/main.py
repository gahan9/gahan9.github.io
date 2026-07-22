"""
Intelligent Defect Triage System - Main Entry Point

Demonstrates the complete RAG pipeline for JIRA defect analysis
and real-time triage prediction.

Usage:
    python -m defect_triage_rag.main --defect-id HPC-1234
    python -m defect_triage_rag.main --demo
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from .embeddings import DefectEmbeddingGenerator
from .models import JiraDefect, TriagePrediction
from .triage_graph import DefectTriageGraph
from .vector_store import DefectVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_demo_defects() -> list[JiraDefect]:
    """Create sample defects for demonstration."""
    return [
        JiraDefect(
            id="HPC-1001",
            summary="GPU memory allocation fails on MI300X under PyTorch DDP",
            description="""
            When running PyTorch Distributed Data Parallel (DDP) training with 8 GPUs
            on MI300X platform, the application crashes with HIP memory allocation error
            after approximately 2 hours of training.
            
            The issue occurs consistently when batch size exceeds 256 per GPU.
            Memory profiling shows gradual increase until OOM.
            """,
            component="ROCm-Memory",
            platform="MI300X",
            error_log="hip_memory_error: failed to allocate 16GB on device 0",
            steps_to_reproduce="1. Launch torchrun with 8 GPUs\n2. Run training for 2+ hours",
        ),
        JiraDefect(
            id="HPC-1002",
            summary="SSH connection timeout during cluster validation",
            description="""
            Automated validation framework fails to connect to compute nodes
            via SSH. Connection times out after 30 seconds. Issue started after
            recent network infrastructure changes.
            """,
            component="Infrastructure",
            platform="MI250X",
            error_log="ssh: connect to host node-01 port 22: Connection timed out",
        ),
        JiraDefect(
            id="HPC-1003",
            summary="HIP kernel launch failure with undefined symbol",
            description="""
            Custom HIP kernel fails to launch with undefined symbol error.
            The kernel was compiled with ROCm 6.0 but the runtime is ROCm 5.7.
            Suspect version mismatch causing symbol resolution failure.
            """,
            component="HIP-Runtime",
            platform="MI300X",
            error_log="hipErrorNotFound: undefined symbol: __hip_fatbin_wrapper",
        ),
        JiraDefect(
            id="HPC-1004",
            summary="Performance regression in GEMM operations after ROCm upgrade",
            description="""
            After upgrading from ROCm 5.7 to ROCm 6.0, observed 15% performance
            regression in GEMM operations. Affects TensorFlow and PyTorch workloads.
            Benchmarking shows increased kernel launch latency.
            """,
            component="rocBLAS",
            platform="MI300X",
        ),
        JiraDefect(
            id="HPC-1005",
            summary="RAS injection test causes system hang",
            description="""
            When injecting correctable memory errors using RAS injection tool,
            the system hangs instead of logging the error and continuing.
            SMI reports the injection but system becomes unresponsive.
            """,
            component="RAS-Firmware",
            platform="MI300X",
            error_log="RAS: Injecting CE at address 0x12345678...[hang]",
        ),
    ]


def initialize_system(
    vector_store_path: Path | None = None,
) -> tuple[DefectEmbeddingGenerator, DefectVectorStore, DefectTriageGraph]:
    """Initialize all system components."""
    
    logger.info("Initializing embedding generator...")
    embedding_gen = DefectEmbeddingGenerator(
        model_name="all-MiniLM-L6-v2",
        device="auto",
    )
    
    logger.info("Initializing vector store...")
    if vector_store_path and vector_store_path.exists():
        vector_store = DefectVectorStore.load(vector_store_path)
    else:
        vector_store = DefectVectorStore(dimension=384)
    
    logger.info("Initializing triage graph...")
    triage_graph = DefectTriageGraph(
        embedding_generator=embedding_gen,
        vector_store=vector_store,
    )
    
    return embedding_gen, vector_store, triage_graph


def index_historical_defects(
    defects: list[JiraDefect],
    embedding_gen: DefectEmbeddingGenerator,
    vector_store: DefectVectorStore,
) -> None:
    """Index historical defects for RAG retrieval."""
    logger.info(f"Indexing {len(defects)} historical defects...")
    
    embeddings = embedding_gen.generate_batch_embeddings(defects)
    vector_store.add_embeddings(embeddings)
    
    logger.info("Indexing complete")
    logger.info(f"Domain distribution: {vector_store.get_domain_statistics()}")


def run_demo() -> None:
    """Run demonstration of the triage system."""
    
    print("\n" + "=" * 60)
    print("  Intelligent Defect Triage System - RAG & LangGraph Demo")
    print("=" * 60 + "\n")
    
    # Initialize system
    embedding_gen, vector_store, triage_graph = initialize_system()
    
    # Create and index demo defects
    demo_defects = create_demo_defects()
    index_historical_defects(demo_defects[:-1], embedding_gen, vector_store)
    
    # Triage a new defect
    new_defect = demo_defects[-1]  # Use last defect as the "new" one
    
    print(f"\n{'─' * 60}")
    print(f"  Triaging Defect: {new_defect.id}")
    print(f"{'─' * 60}")
    print(f"  Summary: {new_defect.summary}")
    print(f"  Component: {new_defect.component}")
    print(f"  Platform: {new_defect.platform}")
    
    # Run triage
    prediction = triage_graph.triage(new_defect)
    
    # Display results
    print(f"\n{'─' * 60}")
    print(f"  Triage Results")
    print(f"{'─' * 60}")
    print(f"  Predicted Domain: {prediction.predicted_domain.value}")
    print(f"  Priority: {prediction.predicted_priority.value}")
    print(f"  Confidence: {prediction.confidence_score:.2%}")
    
    if prediction.predicted_rejection_reason:
        print(f"  Rejection Reason: {prediction.predicted_rejection_reason.value}")
    
    print(f"  Similar Defects: {', '.join(prediction.similar_defects) or 'None'}")
    print(f"  Requires Review: {'Yes' if prediction.requires_manual_review else 'No'}")
    
    print(f"\n  Context Summary:")
    print(f"  {prediction.context_summary[:200]}...")
    
    if prediction.suggested_resolution:
        print(f"\n  Suggested Resolution:")
        print(f"  {prediction.suggested_resolution}")
    
    print("\n" + "=" * 60 + "\n")


def triage_from_json(json_path: Path) -> TriagePrediction:
    """Load defect from JSON and run triage."""
    
    with open(json_path) as f:
        data = json.load(f)
    
    defect = JiraDefect(**data)
    
    embedding_gen, vector_store, triage_graph = initialize_system()
    prediction = triage_graph.triage(defect)
    
    return prediction


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligent Defect Triage System using RAG & LangGraph"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration with sample defects",
    )
    parser.add_argument(
        "--defect-json",
        type=Path,
        help="Path to defect JSON file for triage",
    )
    parser.add_argument(
        "--vector-store",
        type=Path,
        help="Path to vector store directory",
    )
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.defect_json:
        prediction = triage_from_json(args.defect_json)
        print(json.dumps(prediction.model_dump(), indent=2, default=str))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
