"""
M4 Delivery Chatbot Orchestrator for dual-agent coordination.

AI Assistant Notes:
- Transforms the M3 multi-department orchestrator for e-commerce delivery
- Routes between ProductRAGAgent and OrderProcessingAgent based on intent
- Integrates conversation memory and function calling capabilities
- Handles seamless agent handoff for conversational commerce
- Maintains order context across agent transitions
"""

from src.utils import observe
from ..agents import AgentResponse
from ..agents.product_rag_agent import ProductRAGAgent
from ..agents.order_agent import OrderProcessingAgent
from ..conversation import ConversationContext
from ..function_calling.router import FunctionRouter
from ..utils import langfuse_client
from ..config import settings
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any
import time
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class M4OrchestratorResponse:
    """Dataclass for M4 orchestrator responses."""
    answer: str
    agent_used: str
    confidence: float
    processing_time: float
    session_id: str
    intent: str
    function_calls_used: List[str]
    order_created: bool
    order_id: Optional[str] = None
    order_number: Optional[str] = None
    conversation_context: Optional[Dict] = None
    metadata: Optional[Dict] = None


class M4DeliveryOrchestrator:
    """
    M4 Delivery Chatbot orchestrator for e-commerce operations.
    Coordinates product searches and order processing with seamless agent handoff.
    """

    def __init__(self, force_rebuild: bool = False, use_persistent: bool = True):
        """
        Initialize the M4 delivery orchestrator.

        Args:
            force_rebuild: Force rebuild all FAISS indices
            use_persistent: Enable/disable persistent storage
        """
        self.function_router = FunctionRouter()
        self.conversation_context = ConversationContext()
        self.product_agent = None
        self.order_agent = None
        self._initialized = False
        self._force_rebuild = force_rebuild
        self._use_persistent = use_persistent

        # Agent configurations
        self.agent_configs = {
            'product_agent': {
                'name': 'Product Assistant',
                'force_rebuild': force_rebuild,
                'use_persistent': use_persistent
            },
            'order_agent': {
                'name': 'Order Assistant'
            }
        }

    async def initialize(self) -> None:
        """
        Initialize the orchestrator and both agents.
        """
        try:
            logger.info("Initializing M4 Delivery Orchestrator...")

            # Initialize product agent
            self.product_agent = ProductRAGAgent(
                name=self.agent_configs['product_agent']['name']
            )
            # Pass configuration to agent's retriever
            if hasattr(self.product_agent, '_retriever'):
                self.product_agent._retriever.force_rebuild = self._force_rebuild
                self.product_agent._retriever.use_persistent_storage = self._use_persistent

            await self.product_agent.initialize()

            # Initialize order agent
            self.order_agent = OrderProcessingAgent(
                name=self.agent_configs['order_agent']['name']
            )
            await self.order_agent.initialize()

            self._initialized = True
            logger.info("M4 Delivery Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing M4 orchestrator: {e}")
            raise

    @observe(name="m4_delivery_chatbot_processing", as_type="agent")
    async def process_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> M4OrchestratorResponse:
        """
        Process a query through the M4 delivery chatbot pipeline.

        Args:
            query: The input query to process
            user_id: Optional user identifier
            session_id: Optional session identifier

        Returns:
            M4OrchestratorResponse with complete results
        """
        if not self._initialized:
            raise RuntimeError(
                "M4 Orchestrator not initialized. Call initialize() first.")

        # Create or get session
        if not session_id:
            session_id = self.conversation_context.get_or_create_session(
                user_id)

        # Add trace metadata
        if hasattr(langfuse_client.client, 'update_current_trace'):
            langfuse_client.client.update_current_trace(
                metadata={
                    "query_length": len(query),
                    "service_name": "henry_bot_M4_delivery",
                    "source": "delivery_chatbot",
                    "user_id": user_id,
                    "session_id": session_id
                }
            )

        # Use dummy trace context for backward compatibility
        trace = langfuse_client.create_trace(
            name="m4_delivery_chatbot_query",
            input=f"Query: {query[:100]}...",
            metadata={
                'user_id': user_id,
                'session_id': session_id
            }
        )

        start_time = time.time()

        try:
            # Process user message and get conversation context
            conversation_data = self.conversation_context.process_user_message(
                session_id=session_id,
                message=query,
                user_id=user_id
            )

            conversation_history = conversation_data.get(
                'conversation_history', [])
            extracted_info = conversation_data.get('extracted_order_info', {})
            function_suggestions = conversation_data.get(
                'function_suggestions', {})

            # Create comprehensive workflow span
            with trace.span(
                name="m4_delivery_workflow",
                input=query,
                metadata={
                    'stage': 'm4_delivery_processing',
                    'session_id': session_id,
                    'user_id': user_id,
                    'conversation_length': len(conversation_history)
                }
            ) as workflow_span:

                # Step 1: Intent Analysis and Agent Selection
                with trace.span(
                    name="intent_analysis_and_routing",
                    input=query,
                    metadata={
                        'stage': 'agent_selection',
                        'function_calling_suggested': function_suggestions.get('use_function_calling', False),
                        'recommended_tools': function_suggestions.get('recommended_tools', []),
                        'primary_intent': function_suggestions.get('primary_intent', 'unknown'),
                        'conversation_length': len(conversation_history)
                    }
                ) as routing_span:

                    agent_selection_start = time.time()
                    selected_agent, intent, routing_confidence = await self._select_agent(
                        query, conversation_history, extracted_info, function_suggestions
                    )
                    agent_selection_time = time.time() - agent_selection_start

                    # Update routing span with results
                    routing_span.update(
                        output=f"Selected agent: {selected_agent}",
                        metadata={
                            'selected_agent': selected_agent,
                            'intent': intent,
                            'routing_confidence': routing_confidence,
                            'agent_selection_time_seconds': agent_selection_time,
                            'function_calling_recommended': function_suggestions.get('use_function_calling', False)
                        }
                    )

                # Step 2: Agent Processing
                with trace.span(
                    name=f"{selected_agent}_processing",
                    input=f"Query: {query[:100]}... -> Agent: {selected_agent}",
                    metadata={
                        'stage': 'agent_processing',
                        'selected_agent': selected_agent,
                        'intent': intent,
                        'has_conversation_context': len(conversation_history) > 0
                    }
                ) as agent_span:

                    agent_start = time.time()
                    agent_response = await self._execute_agent(
                        selected_agent, query, conversation_history, session_id, trace
                    )
                    agent_execution_time = time.time() - agent_start

                    # Update agent span
                    agent_span.update(
                        output=agent_response.answer[:200] + "..." if len(
                            agent_response.answer) > 200 else agent_response.answer,
                        metadata={
                            'agent_execution_time_seconds': agent_execution_time,
                            'agent_confidence': agent_response.confidence,
                            'agent_metadata': agent_response.metadata
                        }
                    )

                    # Log agent execution
                    langfuse_client.log_agent_execution(
                        trace=trace,
                        agent_name=selected_agent,
                        agent_type="delivery_agent",
                        input_data=query,
                        output_data=agent_response.answer,
                        execution_time=agent_execution_time,
                        metadata={
                            **agent_response.metadata,
                            'intent': intent,
                            'session_id': session_id
                        }
                    )

            # Add assistant response to conversation memory
            self.conversation_context.add_assistant_response(
                session_id=session_id,
                response=agent_response.answer,
                user_id=user_id,
                metadata={
                    'agent_used': selected_agent,
                    'intent': intent,
                    'confidence': agent_response.confidence
                }
            )

            # Calculate total processing time
            total_time = time.time() - start_time

            # Extract order information from response metadata
            order_id = agent_response.metadata.get('order_id')
            order_number = agent_response.metadata.get('order_number')
            order_created = order_id is not None

            # Get updated conversation context
            updated_context = self.conversation_context.get_conversation_summary(
                session_id)

            # Create orchestrator response
            orchestrator_response = M4OrchestratorResponse(
                answer=agent_response.answer,
                agent_used=selected_agent,
                confidence=agent_response.confidence,
                processing_time=total_time,
                session_id=session_id,
                intent=intent,
                function_calls_used=agent_response.metadata.get(
                    'tools_used', []),
                order_created=order_created,
                order_id=order_id,
                order_number=order_number,
                conversation_context=updated_context,
                metadata={
                    **agent_response.metadata,
                    'routing_confidence': routing_confidence,
                    'agent_selection_time': agent_selection_time,
                    'agent_execution_time': agent_execution_time,
                    'conversation_length': len(conversation_history),
                    'extracted_info': extracted_info
                }
            )

            # Update trace with final output
            if trace:
                trace.update(
                    output=orchestrator_response.answer,
                    metadata={
                        'agent_used': orchestrator_response.agent_used,
                        'confidence': orchestrator_response.confidence,
                        'processing_time': orchestrator_response.processing_time,
                        'intent': orchestrator_response.intent,
                        'order_created': orchestrator_response.order_created,
                        'order_id': orchestrator_response.order_id,
                        'session_id': orchestrator_response.session_id,
                        'function_calls_used': orchestrator_response.function_calls_used
                    }
                )

            return orchestrator_response

        except Exception as e:
            error_time = time.time() - start_time
            error_msg = f"Error processing M4 delivery query: {e}"
            logger.error(error_msg)

            # Log error with Langfuse
            if trace:
                langfuse_client.log_error(
                    trace=trace,
                    error_message=error_msg,
                    error_type="orchestrator_processing_error",
                    context={
                        'query': query,
                        'session_id': session_id,
                        'processing_time': error_time
                    }
                )

            # Return error response
            return M4OrchestratorResponse(
                answer="I apologize, but I encountered an error while processing your request. Please try again or contact customer support for assistance.",
                agent_used="error_handler",
                confidence=0.0,
                processing_time=error_time,
                session_id=session_id,
                intent="error",
                function_calls_used=[],
                order_created=False,
                metadata={'error': str(e)}
            )

    async def _select_agent(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        extracted_info: Dict[str, Any],
        function_suggestions: Dict[str, Any]
    ) -> Tuple[str, str, float]:
        """
        Select the appropriate agent based on query and context.

        Args:
            query: User query
            conversation_history: Previous conversation messages
            extracted_info: Extracted order information
            function_suggestions: Function calling suggestions

        Returns:
            Tuple of (selected_agent, intent, confidence)
        """
        # Use function router for intelligent routing
        routing_suggestion = self.function_router.get_tool_suggestions(
            query, conversation_history)

        intent = routing_suggestion.get('primary_intent', 'general_inquiry')
        confidence = routing_suggestion.get('confidence', 0.0)
        use_function_calling = routing_suggestion.get(
            'use_function_calling', False)

        # Only route to Order Agent for EXPLICIT order completion intent
        # Keep all shopping/browsing in Product Agent to close sales first
        order_intents = ['order_status', 'order_update', 'order_cancellation']
        if intent in order_intents and confidence > 0.4:
            return 'order_agent', intent, confidence

        # Only route to Order Agent for EXPLICIT purchase completion language
        # Must include both high intent AND explicit buy/checkout language
        explicit_purchase_keywords = ['checkout', 'add to cart', 'buy now',
                                      'purchase now', 'i want to buy now', 'complete my order']
        if intent == 'order_creation' and confidence > 0.8 and any(keyword in query.lower() for keyword in explicit_purchase_keywords):
            return 'order_agent', 'order_creation', confidence

        # Function calling with order-specific tools -> Order Agent
        if use_function_calling:
            recommended_tools = routing_suggestion.get('recommended_tools', [])
            order_tools = ['create_order', 'check_order_status',
                           'update_order', 'cancel_order']
            if any(tool in order_tools for tool in recommended_tools):
                return 'order_agent', intent, confidence + 0.2

        # Product search intent -> Product Agent
        if intent == 'product_search' and confidence > 0.3:
            return 'product_agent', intent, confidence

        # Check conversation context for order progression
        if conversation_history:
            recent_messages = conversation_history[-3:]
            # If recently discussing products and showing strong purchase intent
            product_discussion = any('product' in msg.get(
                'content', '').lower() for msg in recent_messages)
            # More specific purchase terms that indicate clear intent to buy now
            purchase_terms = ['buy now', 'order now', 'purchase',
                              'add to cart', 'checkout', 'i want to buy']
            purchase_intent = any(term in msg.get('content', '').lower(
            ) for msg in recent_messages for term in purchase_terms)

            # Also check for email addresses in purchase context
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            email_provided = bool(re.findall(email_pattern, query))

            if (product_discussion and purchase_intent) or (product_discussion and email_provided):
                return 'order_agent', 'order_creation', 0.6

        # Default to product agent for general inquiries
        return 'product_agent', intent or 'general_inquiry', confidence or 0.3

    async def _execute_agent(
        self,
        agent_name: str,
        query: str,
        conversation_history: List[Dict[str, str]],
        session_id: str,
        trace
    ) -> AgentResponse:
        """
        Execute the selected agent.

        Args:
            agent_name: Name of the agent to execute
            query: User query
            conversation_history: Conversation context
            session_id: Session identifier
            trace: Langfuse trace

        Returns:
            Agent response
        """
        if agent_name == 'product_agent':
            return await self.product_agent.process_query(query, trace)
        elif agent_name == 'order_agent':
            return await self.order_agent.process_query(query, conversation_history, trace)
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

    def is_initialized(self) -> bool:
        """Check if the orchestrator is initialized."""
        return self._initialized

    def get_available_agents(self) -> List[str]:
        """Get list of available agents."""
        return ['product_agent', 'order_agent']

    async def health_check(self) -> Dict[str, any]:
        """Perform health check on all components."""
        if not self._initialized:
            return {'status': 'not_initialized', 'components': {}}

        health_results = {
            'status': 'healthy',
            'components': {
                'product_agent': self.product_agent.is_initialized() if self.product_agent else False,
                'order_agent': self.order_agent.is_initialized() if self.order_agent else False,
                'function_router': True,  # Always healthy (stateless)
                'conversation_context': True  # Always healthy (has fallbacks)
            }
        }

        # Overall status
        all_healthy = all(health_results['components'].values())
        health_results['status'] = 'healthy' if all_healthy else 'degraded'

        return health_results

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        return self.conversation_context.get_conversation_summary(session_id)

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        return self.conversation_context.cleanup_expired_sessions()
