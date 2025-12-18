"""
RAG agent implementation for specialized department responses.

AI Assistant Notes:
- This is the core RAG implementation that combines retrieval with LLM generation
- Key workflow: retrieve docs → generate context → call LLM → return response
- All operations are fully traced with Langfuse for debugging
- Uses FAISS for fast semantic similarity search
- Confidence scoring based on retrieval quality and response characteristics
- Department-specific system prompts ensure domain-appropriate responses
"""
from ..utils import langfuse_client
from ..config import settings
from ..retrievers import FAISSRetriever, RetrievedDocument
from .base import BaseAgent, AgentResponse
from src.utils import observe
from openai import OpenAI
from typing import List, Dict, Any, Optional
import time


class RAGAgent(BaseAgent):
    """
    Retrieval-Augmented Generation agent specialized for a specific department.
    Combines document retrieval with LLM generation for accurate, context-aware responses.
    """

    def __init__(
        self,
        name: str,
        department: str,
        documents_path: str,
        model_name: str = None,
        similarity_top_k: int = None
    ):
        """
        Initialize RAG agent with configuration.

        Args:
            name: Name of the agent
            department: Department specialization
            documents_path: Path to department documents
            model_name: LLM model name
            similarity_top_k: Number of documents to retrieve
        """
        super().__init__(
            name=name,
            department=department,
            model_name=model_name or settings.model_name
        )

        self.documents_path = documents_path
        self.similarity_top_k = similarity_top_k or settings.similarity_top_k

        # Initialize retriever with persistent storage support
        self._retriever = FAISSRetriever(
            embedding_model=settings.embedding_model,
            similarity_top_k=self.similarity_top_k,
            department_name=department,
            use_persistent_storage=settings.use_persistent_storage,
            force_rebuild=settings.force_rebuild_indices
        )

        # Initialize OpenAI client for OpenRouter
        self._llm_client = None

    async def initialize(self) -> None:
        """
        Initialize the agent and its components.
        """
        try:
            print(
                f"Initializing {self.name} agent for {self.department} department...")

            # Initialize retriever
            await self._retriever.initialize(self.documents_path)

            # Initialize LLM client
            base_url = settings.openrouter_base_url or "https://openrouter.ai/api/v1"
            self._llm_client = OpenAI(
                api_key=settings.openrouter_api_key,
                base_url=base_url,
                model=settings.model_name
            )

            self._initialized = True
            print(f"{self.name} agent initialized successfully")

        except Exception as e:
            print(f"Error initializing {self.name} agent: {e}")
            raise

    @observe(name="rag_agent_processing")
    async def process_query(self, query: str, trace=None) -> AgentResponse:
        """
        Process a query and return a specialized response.
        @observe decorator automatically creates a nested observation.

        Args:
            query: The input query to process
            trace: Optional Langfuse trace for backward compatibility

        Returns:
            AgentResponse with answer, confidence, and metadata
        """
        if not self._initialized:
            raise RuntimeError(
                f"Agent {self.name} not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            # Create comprehensive RAG processing span
            if trace:
                with trace.span(
                    name="rag_pipeline",
                    input=query,
                    metadata={
                        'stage': 'rag_processing',
                        'agent_name': self.name,
                        'department': self.department,
                        'embedding_model': settings.embedding_model
                    }
                ) as rag_pipeline_span:

                    # Document Retrieval with span
                    with trace.span(
                        name="document_retrieval",
                        input=query,
                        metadata={
                            'stage': 'retrieval',
                            'similarity_top_k': self.similarity_top_k,
                            'retriever_type': 'faiss'
                        }
                    ) as retrieval_span:

                        retrieval_start = time.time()
                        retrieved_docs = await self._retriever.retrieve(query)
                        retrieval_time = time.time() - retrieval_start

                        # Update retrieval span with results
                        retrieval_span.update(
                            output=f"Retrieved {len(retrieved_docs)} documents",
                            metadata={
                                'num_documents_retrieved': len(retrieved_docs),
                                'retrieval_time_seconds': retrieval_time,
                                'similarity_scores': [doc.similarity_score for doc in retrieved_docs],
                                'avg_similarity': sum(doc.similarity_score for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
                            }
                        )

                        # Also log as event for detailed metrics
                        langfuse_client.log_rag_retrieval(
                            trace=trace,
                            query=query,
                            retrieved_docs=[
                                doc.content for doc in retrieved_docs],
                            similarity_scores=[
                                doc.similarity_score for doc in retrieved_docs],
                            retrieval_time=retrieval_time
                        )

                    # Context Generation with span
                    with trace.span(
                        name="context_generation",
                        input=f"Query + {len(retrieved_docs)} documents",
                        metadata={
                            'stage': 'prompt_construction',
                            'context_length': sum(len(doc.content) for doc in retrieved_docs),
                            'num_context_docs': len(retrieved_docs)
                        }
                    ) as context_span:

                        contextual_prompt = await self._generate_contextual_prompt(query, retrieved_docs)

                        context_span.update(
                            output=f"Generated prompt of {len(contextual_prompt)} characters",
                            metadata={
                                'prompt_length': len(contextual_prompt),
                                'context_docs_used': len(retrieved_docs)
                            }
                        )

                    # LLM Generation with span
                    with trace.span(
                        name="llm_generation",
                        input=contextual_prompt[:500] + "..." if len(
                            contextual_prompt) > 500 else contextual_prompt,
                        metadata={
                            'stage': 'response_generation',
                            'model_name': self.model_name,
                            'agent_department': self.department
                        }
                    ) as llm_span:

                        llm_start = time.time()
                        llm_response = await self._call_llm(contextual_prompt, trace)
                        llm_time = time.time() - llm_start

                        llm_span.update(
                            output=llm_response[:200] + "..." if len(
                                llm_response) > 200 else llm_response,
                            metadata={
                                'llm_response_time_seconds': llm_time,
                                'response_length': len(llm_response),
                                'model_used': self.model_name
                            }
                        )

                    # Update the main RAG pipeline span
                    rag_pipeline_span.update(
                        output=llm_response[:300] +
                        "..." if len(llm_response) > 300 else llm_response,
                        metadata={
                            'total_pipeline_time_seconds': time.time() - start_time,
                            'retrieval_time_seconds': retrieval_time,
                            'llm_time_seconds': llm_time,
                            'documents_used': len(retrieved_docs)
                        }
                    )
            else:
                # Fallback processing without tracing
                retrieval_start = time.time()
                retrieved_docs = await self._retriever.retrieve(query)
                retrieval_time = time.time() - retrieval_start

                contextual_prompt = await self._generate_contextual_prompt(query, retrieved_docs)

                llm_start = time.time()
                llm_response = await self._call_llm(contextual_prompt, trace)
                llm_time = time.time() - llm_start

            # Calculate confidence based on retrieval scores and response quality
            confidence = self._calculate_confidence(
                retrieved_docs, llm_response)

            # Create response
            response = AgentResponse(
                answer=llm_response,
                confidence=confidence,
                source_documents=[{
                    'content': doc.content,
                    'source': doc.source,
                    'similarity_score': doc.similarity_score,
                    'metadata': doc.metadata
                } for doc in retrieved_docs],
                metadata={
                    'department': self.department,
                    'agent_name': self.name,
                    'retrieval_time': retrieval_time,
                    'llm_time': llm_time,
                    'total_time': time.time() - start_time,
                    'num_retrieved_docs': len(retrieved_docs),
                    'retrieval_success': len(retrieved_docs) > 0
                }
            )

            # Log agent execution with Langfuse
            if trace:
                langfuse_client.log_agent_execution(
                    trace=trace,
                    agent_name=self.name,
                    agent_type=f"rag_{self.department}",
                    input_data=query,
                    output_data=llm_response,
                    execution_time=time.time() - start_time,
                    metadata=response.metadata
                )

            return response

        except Exception as e:
            error_msg = f"Error processing query in {self.name}: {e}"

            # Log error with Langfuse
            if trace:
                langfuse_client.log_error(
                    trace=trace,
                    error_message=error_msg,
                    error_type="agent_processing_error",
                    context={'agent_name': self.name,
                             'department': self.department}
                )

            # Return error response
            return AgentResponse(
                answer="I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists.",
                confidence=0.0,
                source_documents=[],
                metadata={
                    'error': str(e),
                    'department': self.department,
                    'agent_name': self.name
                }
            )

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.

        Returns:
            System prompt string
        """
        department_prompts = {
            'hr': """You are a helpful HR assistant specializing in company policies, benefits, and employee procedures.
            Your role is to provide accurate, helpful information about:
            - Employee benefits and insurance
            - Company policies and procedures
            - Time off and leave policies
            - Workplace conduct and compliance
            - Training and development opportunities

            Always be professional, empathetic, and provide practical guidance based on company policies.""",

            'tech': """You are a knowledgeable IT support specialist helping employees with technical issues.
            Your role is to provide assistance with:
            - Software installation and troubleshooting
            - Hardware and device support
            - Network connectivity and VPN issues
            - Security best practices and procedures
            - Development tools and environments

            Always be technical, precise, and provide step-by-step solutions when possible.""",

            'finance': """You are a finance and expense management specialist helping employees with financial procedures and resources.
            Your role is to provide guidance on:
            - Expense reporting and reimbursement procedures
            - Company budget management and approvals
            - Purchase orders and procurement processes
            - Financial policies and compliance requirements
            - Payroll and compensation procedures
            - Employee financial benefits and resources available
            - Financial planning tools and counseling offered by the company

            When employees ask about personal finance topics (like personal budgeting):
            1. Focus on company-provided resources and benefits first
            2. Reference Employee Assistance Programs, financial counseling, and workshops
            3. Provide information about company financial benefits that support personal financial goals
            4. Use company procedures and policies as examples where relevant
            5. Acknowledge limitations of company documents for personal advice
            6. Be accurate, detail-oriented, and provide clear procedural guidance"""
        }

        return department_prompts.get(
            self.department,
            "You are a helpful company assistant. Provide accurate, professional assistance with employee inquiries."
        )

    async def _call_llm(self, prompt: str, trace=None) -> str:
        """
        Call the LLM with the given prompt.

        Args:
            prompt: The prompt to send to the LLM
            trace: Optional Langfuse trace for observability

        Returns:
            LLM response string
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized")

        start_time = time.time()
        try:
            response = self._llm_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/estebmaister/henry_bot_M4",
                    "X-Title": "henry_bot_M4-RAGAgent"
                },
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3,
            )

            response_time = time.time() - start_time
            content = response.choices[0].message.content.strip()

            # Log LLM call with enhanced debugging
            if trace:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }

                langfuse_client.log_llm_call(
                    trace=trace,
                    model_name=self.model_name,
                    prompt=prompt,
                    response=content,
                    token_usage=token_usage,
                    response_time=response_time,
                    temperature=0.3,
                    max_tokens=1000,
                    metadata={
                        'department': self.department,
                        'agent_name': self.name,
                        'model_provider': 'openrouter',
                        'response_finish_reason': response.choices[0].finish_reason if response.choices else None,
                        'response_id': response.id
                    }
                )

            return content

        except Exception as e:
            error_time = time.time() - start_time
            error_msg = f"Error calling LLM: {e}"

            # Log LLM error
            if trace:
                langfuse_client.log_error(
                    trace=trace,
                    error_message=error_msg,
                    error_type="llm_call_error",
                    context={
                        'model_name': self.model_name,
                        'department': self.department,
                        'error_time_seconds': error_time,
                        'prompt_preview': prompt[:200] if prompt else None
                    }
                )

            raise

    def _calculate_confidence(self, retrieved_docs: List[RetrievedDocument], response: str) -> float:
        """
        Calculate confidence score based on retrieval and response quality.

        Args:
            retrieved_docs: List of retrieved documents
            response: Generated response

        Returns:
            Confidence score between 0 and 1
        """
        if not retrieved_docs:
            return 0.3  # Low confidence without context

        # Base confidence from retrieval scores
        retrieval_confidence = sum(
            doc.similarity_score for doc in retrieved_docs) / len(retrieved_docs)

        # Adjust for response length (very short responses might indicate uncertainty)
        # Normalize to 200 characters
        response_length_factor = min(1.0, len(response) / 200)

        # Combine factors
        confidence = (retrieval_confidence * 0.7) + \
            (response_length_factor * 0.3)

        return min(1.0, max(0.0, confidence))

    async def _generate_contextual_prompt(self, query: str, retrieved_docs: List[RetrievedDocument]) -> str:
        """
        Generate a contextual prompt using retrieved documents.

        Args:
            query: The original user query
            retrieved_docs: List of retrieved documents

        Returns:
            Contextual prompt for LLM
        """
        # Combine system prompt with retrieved context
        system_prompt = self.get_system_prompt()

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"Document {i} (Source: {doc.source}):\n{doc.content}")

        context_text = "\n\n".join(
            context_parts) if context_parts else "No specific documents were found for this query."

        # Create the full prompt
        full_prompt = f"""{system_prompt}

CONTEXT INFORMATION:
{context_text}

USER QUERY: {query}

INSTRUCTIONS:
1. Based on the provided context information, answer the user's query accurately and comprehensively
2. Use specific details, procedures, and resources mentioned in the documents
3. If the context doesn't fully address the user's specific need, acknowledge this limitation
4. Prioritize information from the provided company documents over general advice
5. If relevant company resources or procedures are mentioned in the context, highlight them
6. Be helpful and professional while staying grounded in the provided documentation

ANSWER:"""

        return full_prompt
