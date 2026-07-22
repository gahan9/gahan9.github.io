"""
LangGraph Triage Workflow

Implements the core RAG pipeline using LangGraph for orchestrating
the defect triage workflow with multiple processing nodes.
"""

import logging
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .embeddings import DefectEmbeddingGenerator
from .models import (
    DefectDomain,
    JiraDefect,
    RejectionReason,
    TriagePrediction,
    TriagePriority,
    TriageState,
)
from .vector_store import DefectVectorStore

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State schema for LangGraph workflow."""
    
    defect: JiraDefect
    embedding: list[float] | None
    retrieved_context: list[dict]
    domain_classification: DefectDomain | None
    trend_analysis: dict
    prediction: TriagePrediction | None
    messages: Annotated[list, add_messages]
    error: str | None


class DefectTriageGraph:
    """
    LangGraph-based RAG pipeline for intelligent defect triage.
    
    Orchestrates the following workflow:
    1. Embed incoming defect
    2. Retrieve similar defects (RAG)
    3. Classify domain
    4. Analyze trends
    5. Generate triage prediction
    """
    
    SYSTEM_PROMPT = """You are an expert HPC validation engineer specializing in 
AMD GPU platforms (MI300X, MI250X). Your task is to analyze defects and provide 
intelligent triage predictions.

Based on the defect information and similar historical defects, determine:
1. The root cause domain (infrastructure, memory, system call, etc.)
2. Priority level (P0-P4)
3. Likely rejection reason if applicable
4. Suggested resolution steps

Be concise and technical in your analysis."""

    def __init__(
        self,
        embedding_generator: DefectEmbeddingGenerator,
        vector_store: DefectVectorStore,
        llm_model: str = "gpt-4-turbo-preview",
    ) -> None:
        """
        Initialize the triage graph.
        
        Args:
            embedding_generator: PyTorch embedding generator.
            vector_store: Vector store for RAG retrieval.
            llm_model: LLM model name for analysis.
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Define the graph
        builder = StateGraph(GraphState)
        
        # Add nodes
        builder.add_node("embed_defect", self._embed_defect)
        builder.add_node("retrieve_context", self._retrieve_context)
        builder.add_node("classify_domain", self._classify_domain)
        builder.add_node("analyze_trends", self._analyze_trends)
        builder.add_node("generate_prediction", self._generate_prediction)
        builder.add_node("handle_error", self._handle_error)
        
        # Add edges
        builder.add_edge(START, "embed_defect")
        builder.add_conditional_edges(
            "embed_defect",
            self._check_embedding_success,
            {
                "success": "retrieve_context",
                "error": "handle_error",
            },
        )
        builder.add_edge("retrieve_context", "classify_domain")
        builder.add_edge("classify_domain", "analyze_trends")
        builder.add_edge("analyze_trends", "generate_prediction")
        builder.add_edge("generate_prediction", END)
        builder.add_edge("handle_error", END)
        
        return builder.compile()
    
    def _embed_defect(self, state: GraphState) -> GraphState:
        """Node: Generate embedding for the defect."""
        try:
            defect = state["defect"]
            embedding_result = self.embedding_generator.generate_embedding(defect)
            
            return {
                **state,
                "embedding": embedding_result.embedding,
                "messages": [
                    HumanMessage(content=f"Analyzing defect: {defect.id}")
                ],
            }
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return {**state, "error": f"Embedding generation failed: {e}"}
    
    def _check_embedding_success(
        self, state: GraphState
    ) -> Literal["success", "error"]:
        """Conditional: Check if embedding was successful."""
        if state.get("error"):
            return "error"
        return "success"
    
    def _retrieve_context(self, state: GraphState) -> GraphState:
        """Node: Retrieve similar defects using RAG."""
        embedding = state["embedding"]
        
        if not embedding:
            return state
        
        # Search for similar defects
        similar_defects = self.vector_store.search(
            query_embedding=embedding,
            top_k=5,
        )
        
        return {
            **state,
            "retrieved_context": similar_defects,
            "messages": [
                AIMessage(
                    content=f"Found {len(similar_defects)} similar defects for context"
                )
            ],
        }
    
    def _classify_domain(self, state: GraphState) -> GraphState:
        """Node: Classify defect domain based on content and similar defects."""
        defect = state["defect"]
        retrieved = state.get("retrieved_context", [])
        
        # Aggregate domain votes from similar defects
        domain_votes: dict[str, float] = {}
        for ctx in retrieved:
            domain = ctx.get("metadata", {}).get("domain", "unknown")
            score = ctx.get("similarity_score", 0.5)
            domain_votes[domain] = domain_votes.get(domain, 0) + score
        
        # Determine most likely domain
        if domain_votes:
            predicted_domain = max(domain_votes, key=domain_votes.get)
            domain_enum = DefectDomain(predicted_domain)
        else:
            # Fallback to keyword-based classification
            emb = self.embedding_generator.generate_embedding(defect)
            domain_enum = emb.domain
        
        return {
            **state,
            "domain_classification": domain_enum,
        }
    
    def _analyze_trends(self, state: GraphState) -> GraphState:
        """Node: Analyze defect trends and patterns."""
        retrieved = state.get("retrieved_context", [])
        domain = state.get("domain_classification")
        
        # Calculate trend statistics
        trend_analysis = {
            "similar_count": len(retrieved),
            "domain": domain.value if domain else "unknown",
            "avg_similarity": (
                sum(r.get("similarity_score", 0) for r in retrieved) / len(retrieved)
                if retrieved else 0
            ),
            "domain_distribution": self.vector_store.get_domain_statistics(),
        }
        
        # Identify if this is a recurring pattern
        if retrieved and trend_analysis["avg_similarity"] > 0.85:
            trend_analysis["pattern_type"] = "recurring"
            trend_analysis["likely_known_issue"] = True
        else:
            trend_analysis["pattern_type"] = "novel"
            trend_analysis["likely_known_issue"] = False
        
        return {
            **state,
            "trend_analysis": trend_analysis,
        }
    
    def _generate_prediction(self, state: GraphState) -> GraphState:
        """Node: Generate final triage prediction using LLM."""
        defect = state["defect"]
        domain = state.get("domain_classification", DefectDomain.UNKNOWN)
        retrieved = state.get("retrieved_context", [])
        trends = state.get("trend_analysis", {})
        
        # Build context for LLM
        context_text = self._build_context_text(defect, retrieved, trends)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", context_text),
        ])
        
        # Get LLM response
        chain = prompt | self.llm
        response = chain.invoke({})
        
        # Parse response and create prediction
        prediction = self._parse_llm_response(
            defect_id=defect.id,
            domain=domain,
            response=response.content,
            retrieved=retrieved,
            trends=trends,
        )
        
        return {
            **state,
            "prediction": prediction,
            "messages": [
                AIMessage(content=f"Triage complete: {prediction.predicted_priority.value}")
            ],
        }
    
    def _build_context_text(
        self,
        defect: JiraDefect,
        retrieved: list[dict],
        trends: dict,
    ) -> str:
        """Build context text for LLM prompt."""
        parts = [
            "## Current Defect",
            f"**ID:** {defect.id}",
            f"**Summary:** {defect.summary}",
            f"**Description:** {defect.description}",
            f"**Component:** {defect.component}",
            f"**Platform:** {defect.platform}",
        ]
        
        if defect.error_log:
            parts.append(f"**Error Log:** {defect.error_log[:500]}")
        
        parts.append("\n## Similar Historical Defects")
        for ctx in retrieved[:3]:
            parts.append(
                f"- {ctx['defect_id']} (similarity: {ctx['similarity_score']:.2f}, "
                f"domain: {ctx['metadata'].get('domain', 'unknown')})"
            )
        
        parts.append("\n## Trend Analysis")
        parts.append(f"- Pattern Type: {trends.get('pattern_type', 'unknown')}")
        parts.append(f"- Likely Known Issue: {trends.get('likely_known_issue', False)}")
        
        parts.append("\n## Task")
        parts.append(
            "Analyze this defect and provide: "
            "1) Priority (P0-P4), "
            "2) Root cause category, "
            "3) Likely rejection reason if applicable, "
            "4) Suggested resolution."
        )
        
        return "\n".join(parts)
    
    def _parse_llm_response(
        self,
        defect_id: str,
        domain: DefectDomain,
        response: str,
        retrieved: list[dict],
        trends: dict,
    ) -> TriagePrediction:
        """Parse LLM response into structured prediction."""
        
        # Determine priority based on keywords and trends
        priority = TriagePriority.MEDIUM
        if "critical" in response.lower() or "p0" in response.lower():
            priority = TriagePriority.CRITICAL
        elif "high" in response.lower() or "p1" in response.lower():
            priority = TriagePriority.HIGH
        elif "low" in response.lower() or "p3" in response.lower():
            priority = TriagePriority.LOW
        
        # Determine rejection reason if applicable
        rejection_reason = None
        if "setup" in response.lower() or "environment" in response.lower():
            rejection_reason = RejectionReason.SETUP_ISSUE
        elif "infrastructure" in response.lower() or "infra" in response.lower():
            rejection_reason = RejectionReason.INFRA_ISSUE
        elif "duplicate" in response.lower():
            rejection_reason = RejectionReason.DUPLICATE
        elif "working as designed" in response.lower():
            rejection_reason = RejectionReason.WORKING_AS_DESIGNED
        
        # Calculate confidence based on context quality
        confidence = min(0.95, 0.5 + (trends.get("avg_similarity", 0) * 0.5))
        
        return TriagePrediction(
            defect_id=defect_id,
            predicted_domain=domain,
            predicted_priority=priority,
            predicted_rejection_reason=rejection_reason,
            confidence_score=confidence,
            similar_defects=[r["defect_id"] for r in retrieved[:5]],
            suggested_resolution=self._extract_resolution(response),
            context_summary=response[:500],
            requires_manual_review=confidence < 0.7,
        )
    
    def _extract_resolution(self, response: str) -> str:
        """Extract suggested resolution from LLM response."""
        # Look for resolution section
        lower_resp = response.lower()
        
        for keyword in ["resolution:", "suggestion:", "fix:", "solution:"]:
            if keyword in lower_resp:
                start = lower_resp.find(keyword)
                end = lower_resp.find("\n", start + 50)
                if end == -1:
                    end = min(start + 200, len(response))
                return response[start:end].strip()
        
        return "Manual analysis recommended"
    
    def _handle_error(self, state: GraphState) -> GraphState:
        """Node: Handle errors in the pipeline."""
        error = state.get("error", "Unknown error")
        logger.error(f"Pipeline error: {error}")
        
        # Create fallback prediction
        defect = state["defect"]
        prediction = TriagePrediction(
            defect_id=defect.id,
            predicted_domain=DefectDomain.UNKNOWN,
            predicted_priority=TriagePriority.MEDIUM,
            confidence_score=0.0,
            context_summary=f"Error during analysis: {error}",
            requires_manual_review=True,
        )
        
        return {
            **state,
            "prediction": prediction,
        }
    
    def triage(self, defect: JiraDefect) -> TriagePrediction:
        """
        Run triage pipeline for a defect.
        
        Args:
            defect: JIRA defect to triage.
            
        Returns:
            Triage prediction with analysis results.
        """
        initial_state: GraphState = {
            "defect": defect,
            "embedding": None,
            "retrieved_context": [],
            "domain_classification": None,
            "trend_analysis": {},
            "prediction": None,
            "messages": [],
            "error": None,
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return result["prediction"]
    
    async def triage_async(self, defect: JiraDefect) -> TriagePrediction:
        """Async version of triage for batch processing."""
        initial_state: GraphState = {
            "defect": defect,
            "embedding": None,
            "retrieved_context": [],
            "domain_classification": None,
            "trend_analysis": {},
            "prediction": None,
            "messages": [],
            "error": None,
        }
        
        result = await self.graph.ainvoke(initial_state)
        return result["prediction"]
