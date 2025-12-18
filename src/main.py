"""
M4 Delivery Chatbot - Main Application Entry Point

AI Assistant Notes:
- Main entry point for the M4 delivery chatbot system
- CLI interface for product searches and order management
- Integrates dual-agent architecture with conversation memory
- Provides comprehensive command-line interface for testing and management
- Maintains backward compatibility with M3 commands
"""

from src.data.seeds.products_seed import ProductSeeder
from src.database.connection import DatabaseConnection
from src.database.migrations import DatabaseMigrations
from src.orchestrator.orchestrator import M4DeliveryOrchestrator
from src.config import settings
import argparse
import asyncio
import sys
import os
import time
import uuid
import logging
from typing import Optional, Dict, Any

# Set environment variables for compatibility
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_NO_GGML'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class M4DeliveryChatbot:
    """
    Main application class for the M4 Delivery Chatbot.
    Handles initialization, command processing, and system management.
    """

    def __init__(self):
        """Initialize the chatbot application."""
        self.orchestrator: Optional[M4DeliveryOrchestrator] = None
        self.db_connection: Optional[DatabaseConnection] = None
        self._cleaned_up = False  # Prevent duplicate cleanup

    def _generate_user_id(self) -> str:
        """Generate a unique user ID for this interactive session."""
        return f"interactive_user_{uuid.uuid4().hex[:12]}_{int(time.time())}"

    async def initialize(self, force_rebuild: bool = False) -> None:
        """
        Initialize the chatbot system.

        Args:
            force_rebuild: Force rebuild all indices and databases
        """
        try:
            logger.info("Initializing M4 Delivery Chatbot...")

            # Initialize database
            await self._initialize_database()

            # Initialize orchestrator
            self.orchestrator = M4DeliveryOrchestrator(
                force_rebuild=force_rebuild,
                use_persistent=settings.use_persistent_storage
            )
            await self.orchestrator.initialize()

            logger.info("M4 Delivery Chatbot initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {e}")
            raise

    async def _initialize_database(self) -> None:
        """Initialize database with migrations."""
        self.db_connection = DatabaseConnection(settings.database_path)
        migrations = DatabaseMigrations(self.db_connection)
        migrations.run_migrations()

    async def process_query(
        self,
        query: str,
        user_id: str = None,
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the chatbot.

        Args:
            query: User query string
            user_id: Optional user identifier
            session_id: Optional session identifier

        Returns:
            Dictionary with response and metadata
        """
        if not self.orchestrator:
            raise RuntimeError(
                "Chatbot not initialized. Call initialize() first.")

        response = await self.orchestrator.process_query(query, user_id, session_id)

        return {
            'query': query,
            'answer': response.answer,
            'agent_used': response.agent_used,
            'confidence': response.confidence,
            'intent': response.intent,
            'processing_time': response.processing_time,
            'session_id': response.session_id,
            'order_created': response.order_created,
            'order_id': response.order_id,
            'order_number': response.order_number,
            'function_calls_used': response.function_calls_used,
            'conversation_context': response.conversation_context
        }

    async def run_interactive_mode(self) -> None:
        """Run the chatbot in interactive mode."""
        # Generate unique user ID for this session
        user_id = self._generate_user_id()

        print("🤖 M4 Delivery Chatbot - Interactive Mode")
        print(f"🔑 Session ID: {user_id}")
        print("💬 Every conversation gets a unique ID for personalization")
        print("Type 'quit', 'exit', or Ctrl+C to exit")
        print("Type 'help' for available commands")
        print("-" * 50)

        current_session_id = None
        shutdown_requested = False
        message_count = 0

        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            shutdown_requested = True
            print("\n\nShutdown signal received. Exiting...")
            # Raise KeyboardInterrupt to break out of input
            raise KeyboardInterrupt()

        try:
            import signal
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except (ValueError, OSError):
            # Signal handling may not be available in all environments
            pass

        try:
            while not shutdown_requested:
                try:
                    # Simple synchronous input
                    query = input("\nYou: ").strip()

                    if shutdown_requested:
                        break

                    if query.lower() in ['quit', 'exit']:
                        print("Goodbye! 👋")
                        break

                    if query.lower() == 'help':
                        self._show_help()
                        continue

                    if query.lower() == 'clear':
                        current_session_id = None
                        print("Session cleared. Starting fresh conversation.")
                        continue

                    if not query:
                        continue

                    message_count += 1

                    # Process query with user ID
                    result = await self.process_query(query, user_id=user_id, session_id=current_session_id)
                    current_session_id = result['session_id']

                    # Validate response content
                    if not result['answer'] or not result['answer'].strip():
                        logger.warning(
                            f"Empty response received for query: {query}")
                        print(f"\n🤖 Assistant ({result['agent_used']}):")
                        print(
                            "I apologize, but I couldn't generate a response. Please try rephrasing your question.")
                    else:
                        # Display response with enhanced formatting
                        print(f"\n🤖 Assistant ({result['agent_used']}):")
                        print(result['answer'])

                    # Show order information if created
                    if result['order_created']:
                        print(f"\n📦 Order Created!")
                        print(f"   Order Number: {result['order_number']}")
                        print(f"   Order ID: {result['order_id']}")

                    # Show session info in debug mode
                    if settings.debug:
                        print(f"\n📊 Session Info:")
                        print(f"   User ID: {user_id}")
                        print(f"   Session ID: {result['session_id']}")
                        print(f"   Message Count: {message_count}")
                        print(f"   Intent: {result['intent']}")
                        print(f"   Confidence: {result['confidence']:.2f}")
                        print(
                            f"   Processing Time: {result['processing_time']:.2f}s")
                        if result['function_calls_used']:
                            print(
                                f"   Function Calls: {', '.join(result['function_calls_used'])}")
                    else:
                        # Show brief session status
                        print(
                            f"\n💬 Message #{message_count} • Session: {result['session_id'][:8]}...")

                except KeyboardInterrupt:
                    print("Goodbye! 👋")
                    break
                except EOFError:
                    # stdin closed
                    print("\nInput closed. Goodbye! 👋")
                    break
                except Exception as e:
                    logger.error(f"Error in interactive mode: {e}")
                    print(f"\n❌ Error: {e}")
                    if not shutdown_requested:
                        print("Continuing... (type 'quit' to exit)")
                    continue

        except Exception as e:
            logger.error(f"Unexpected error in interactive mode: {e}")
        finally:
            logger.info("Interactive mode ended")

            # Show session summary if messages were exchanged
            if message_count > 0:
                print(f"\n📊 Session Summary:")
                print(f"   Total Messages: {message_count}")
                print(f"   Session ID: {user_id}")
                if current_session_id:
                    print(f"   Conversation ID: {current_session_id[:8]}...")
                print(f"   Thank you for using M4 Delivery Chatbot! 👋")

    def _show_help(self) -> None:
        """Show help information."""
        help_text = """
🤖 M4 Delivery Chatbot Commands:

General Commands:
  help     - Show this help message
  clear    - Clear conversation history
  quit     - Exit the chatbot

Sample Queries:
  Product Search:
    "Find me an iPhone 15"
    "What laptops do you have under $1000?"
    "Show me Bluetooth speakers"

  Order Creation:
    "I want to buy the iPhone 15 Pro"
    "I'll take 2 PlayStation 5 consoles"
    "Add the Samsung TV to my order"

  Order Management:
    "Check order status ORD-123456"
    "What's the status of my order?"
    "I need to cancel my order"

Conversation Features:
  - Multi-turn conversations with context
  - Automatic order extraction from chat history
  - Seamless handoff between product search and ordering
  - Real-time inventory and pricing information
        """
        print(help_text)

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._cleaned_up:
            return  # Prevent duplicate cleanup

        try:
            logger.info("Starting M4 Delivery Chatbot cleanup...")

            # Cancel any pending tasks
            try:
                tasks = [task for task in asyncio.all_tasks(
                ) if not task.done() and task != asyncio.current_task()]
                if tasks:
                    logger.info(f"Cancelling {len(tasks)} background tasks...")
                    for task in tasks:
                        task.cancel()

                    # Wait for tasks to cancel with timeout
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=5.0
                    )
            except Exception as e:
                logger.warning(f"Error cancelling background tasks: {e}")

            if self.orchestrator:
                # Cleanup expired sessions
                try:
                    cleaned = self.orchestrator.cleanup_expired_sessions()
                    if cleaned > 0:
                        logger.info(f"Cleaned up {cleaned} expired sessions")
                except Exception as e:
                    logger.warning(f"Error cleaning up sessions: {e}")

            # Close database connection
            if self.db_connection:
                try:
                    self.db_connection.close_all_connections()
                    logger.info("Database connection closed")
                except Exception as e:
                    logger.warning(f"Error closing database connection: {e}")

            logger.info("M4 Delivery Chatbot shutdown complete")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            self._cleaned_up = True  # Mark as cleaned up


async def main():
    """Main entry point for the M4 delivery chatbot."""
    parser = argparse.ArgumentParser(description="M4 Delivery Chatbot")
    parser.add_argument(
        'command',
        choices=['interactive', 'query', 'init', 'seed', 'status'],
        help='Command to execute'
    )
    parser.add_argument('--query', type=str, help='Query to process')
    parser.add_argument('--user-id', type=str, help='User ID for tracing')
    parser.add_argument('--session-id', type=str,
                        help='Session ID for conversation')
    parser.add_argument('--force-rebuild', action='store_true',
                        help='Force rebuild all indices')
    parser.add_argument('--file', type=str,
                        help='Test file for batch processing')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    args = parser.parse_args()

    # Override debug setting if provided
    if args.debug:
        settings.debug = True
        logging.getLogger().setLevel(logging.DEBUG)

    chatbot = None

    try:
        chatbot = M4DeliveryChatbot()

        if args.command == 'interactive':
            await chatbot.initialize(force_rebuild=args.force_rebuild)
            try:
                await chatbot.run_interactive_mode()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye! 👋")
                # Gracefully handle Ctrl+C and EOF without traceback
                return

        elif args.command == 'query':
            await chatbot.initialize(force_rebuild=args.force_rebuild)
            if not args.query:
                print("Error: --query is required for query command")
                sys.exit(1)

            result = await chatbot.process_query(
                args.query,
                user_id=args.user_id,
                session_id=args.session_id
            )

            print(f"Query: {result['query']}")
            print(f"Answer: {result['answer']}")
            print(f"Agent: {result['agent_used']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Processing Time: {result['processing_time']:.2f}s")

            if result['order_created']:
                print(f"Order Created: {result['order_number']}")

        elif args.command == 'init':
            await chatbot.initialize(force_rebuild=args.force_rebuild)
            print("M4 Delivery Chatbot initialized successfully!")

        elif args.command == 'seed':
            # Seed product database
            await chatbot._initialize_database()
            seeder = ProductSeeder()
            seeder.setup_database()
            results = seeder.seed_all_products()

            print("Product database seeded successfully!")
            for category, count in results.items():
                print(f"  {category}: {count} products")

            summary = seeder.get_seeding_summary()
            print(f"\nDatabase Summary:")
            print(f"  Total Products: {summary['total_products']}")
            print(f"  Total Value: ${summary['total_inventory_value']:,.2f}")

        elif args.command == 'status':
            await chatbot.initialize()
            health = await chatbot.orchestrator.health_check()

            print("M4 Delivery Chatbot Status:")
            print(f"  Overall: {health['status']}")
            print("  Components:")
            for component, status in health['components'].items():
                status_icon = "✅" if status else "❌"
                print(
                    f"    {status_icon} {component}: {'Healthy' if status else 'Unhealthy'}")

    except (KeyboardInterrupt, EOFError):
        # Handle Ctrl+C and EOF gracefully without traceback
        print("\nGoodbye! 👋")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if chatbot:
            try:
                await chatbot.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    asyncio.run(main())
