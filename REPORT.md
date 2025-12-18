# Henry Bot M4 Report

## Overview
This report documents the architecture, functionality, and components of the henry_bot_M4 repository. The M4 Delivery Chatbot is a sophisticated conversational e-commerce system that combines Retrieval-Augmented Generation (RAG) with autonomous function calling for intelligent product discovery and order management.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Configuration](#configuration)
6. [Database Schema](#database-schema)
7. [Function Calling & Tools](#function-calling--tools)
8. [Conversation Management](#conversation-management)
9. [Quality Assessment & Evaluation](#quality-assessment--evaluation)
10. [Observability & Monitoring](#observability--monitoring)
11. [Data Sources & Embeddings](#data-sources--embeddings)
12. [Dependencies](#dependencies)
13. [Usage Examples](#usage-examples)

---

## Project Overview

### System Purpose
The M4 Delivery Chatbot is an intelligent conversational commerce platform that transforms the M3 multi-department HR system into a sophisticated e-commerce delivery assistant. It enables natural language product discovery, order creation, and customer service through a dual-agent architecture.

### Key Capabilities
- **Semantic Product Search**: FAISS-based vector retrieval with 32+ products across 4 categories
- **Autonomous Order Processing**: OpenAI Function Calling with 5 specialized tools
- **Conversational Intelligence**: Multi-turn conversations with context memory
- **Real-time Inventory**: Live stock levels and pricing integration
- **Production-Ready**: Complete database persistence, validation, and error handling

### Business Value
- **Sales Conversion**: Seamless transition from product discovery to order completion
- **Customer Experience**: Natural language interface with intelligent recommendations
- **Operational Efficiency**: Automated order processing with minimal human intervention
- **Scalability**: Modular architecture supporting easy extension and maintenance

---

## Architecture

### High-Level Design
```
┌─────────────────────────────────────────────────────────┐
│                M4 Delivery Chatbot                      │
├─────────────────────────────────────────────────────────┤
│  ┌───────────────────┐    ┌─────────────────────────┐   │
│  │   Orchestrator    │───▶│  Function Router        │   │
│  │ (Dual-Agent Mgr)  │    │  (Intent Analysis)      │   │
│  └───────────────────┘    └─────────────────────────┘   │
└─────────────────┬───────────────────────────────────────┘
                  │
        ┌─────────┼─────────────────┐
        │         │                 │
        ▼         ▼                 ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────────┐
│ProductRAG   │ │Order        │ │Conversation     │
│Agent        │ │Processing   │ │Memory           │
│             │ │Agent        │ │                 │
└─────────────┘ └─────────────┘ └─────────────────┘
        │             │                    │
        ▼             ▼                    ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────────┐
│Vector Store │ │Function     │ │SQLite Database   │
│+ Products   │ │Calling      │ │(Orders, Sessions)│
│             │ │Engine       │ │                  │
└─────────────┘ └─────────────┘ └──────────────────┘
```

### Architectural Patterns
- **Dual-Agent System**: Specialized agents for product discovery and order processing
- **Intelligent Orchestrator**: Context-aware routing with seamless agent handoff
- **RAG Integration**: Vector search combined with real-time database queries
- **Function Calling**: Autonomous tool execution for order management
- **Conversation Memory**: Sliding window with persistent session storage

---

## Core Components

### 1. M4DeliveryOrchestrator (`src/orchestrator/orchestrator.py`)
**Role**: Central coordination and intelligent routing
**Key Features**:
- Dual-agent coordination and lifecycle management
- Intent-based agent selection with confidence scoring
- Context-aware conversation routing
- Comprehensive error handling and fallbacks
- Health monitoring and session management

**Key Methods**:
- `process_query()`: Main query processing pipeline
- `_select_agent()`: Intelligent agent routing based on intent
- `_execute_agent()`: Agent execution with context passing

### 2. ProductRAGAgent (`src/agents/product_rag_agent.py`)
**Role**: Semantic product discovery and recommendation
**Key Features**:
- FAISS-based vector search with product embeddings
- Real-time inventory and pricing integration
- Enhanced product context with database enrichment
- Sales-oriented response generation
- Multi-category product support (Electronics, Books, Clothing, Home Goods)

**Key Methods**:
- `process_query()`: Product search with RAG pipeline
- `_enhance_products_with_db_data()`: Real-time data integration
- `_generate_product_contextual_prompt()`: Context-aware prompt generation

### 3. OrderProcessingAgent (`src/agents/order_agent.py`)
**Role**: Autonomous order management via Function Calling
**Key Features**:
- OpenAI Function Calling with 5 specialized tools
- Comprehensive order CRUD operations
- Simplified checkout process with email-based payment
- Sales completion and order validation
- Graceful fallbacks for Claude models

**Key Methods**:
- `process_query()`: Function calling orchestration
- `_execute_function_calls()`: Tool execution and result processing
- `create_order_simplified()`: Direct order creation for fallback scenarios

### 4. FAISSRetriever (`src/retrievers/faiss_retriever.py`)
**Role**: Vector similarity search for product discovery
**Key Features**:
- Persistent storage with department-specific indices
- Cosine similarity search with configurable top-k
- Efficient embedding generation and caching
- Dynamic document addition and index updates

**Key Methods**:
- `retrieve()`: Semantic search with similarity scoring
- `initialize()`: Index building with persistent storage support
- `add_documents()`: Dynamic document addition

### 5. ConversationContext (`src/conversation/context.py`)
**Role**: Conversation state management and context extraction
**Key Features**:
- Sliding window memory (20 messages)
- Session persistence with timeout management
- Order information extraction from conversation history
- Function calling opportunity analysis

**Key Methods**:
- `process_user_message()`: Message processing with context update
- `get_conversation_summary()`: Comprehensive session analysis
- `cleanup_expired_sessions()`: Session lifecycle management

---

## Data Flow

### Query Processing Pipeline
1. **Input Reception**: User query received via CLI or API
2. **Context Processing**: Conversation history analysis and entity extraction
3. **Intent Analysis**: Function routing determines user intent and agent selection
4. **Agent Execution**: Specialized agent processes query with relevant tools
5. **Response Generation**: Contextual response with appropriate actions
6. **Context Update**: Conversation memory updated with interaction
7. **Observability**: Comprehensive tracing and logging

### Agent Handoff Scenarios
- **Product → Order**: When purchase intent detected in product discussion
- **Order → Product**: When additional product information needed during ordering
- **Multi-turn Support**: Context maintained across extended conversations

### Error Handling Flow
1. **Agent-level Fallbacks**: Individual agent error recovery
2. **Orchestrator Intervention**: Cross-agent error handling
3. **Graceful Degradation**: User-friendly error responses
4. **Comprehensive Logging**: Detailed error tracking for debugging

---

## Configuration

### Settings Management (`src/config.py`)
Uses Pydantic Settings for type-safe configuration with environment variable support.

### Key Configuration Categories

#### LLM Configuration
```python
openrouter_api_key: str
model_name: str = "anthropic/claude-3.5-haiku"
function_calling_model: str = "gpt-4"
```

#### RAG Configuration
```python
embedding_model: str = "all-MiniLM-L6-v2"
similarity_top_k: int = 3
chunk_size: int = 1000
chunk_overlap: int = 200
```

#### Database Configuration
```python
database_path: str = "database/orders.db"
database_pool_size: int = 5
database_timeout: int = 30
```

#### Business Logic
```python
low_stock_threshold: int = 5
max_items_per_order: int = 100
tax_rate: float = 0.08
shipping_cost_threshold: float = 50.00
```

#### Conversation Memory
```python
conversation_memory_size: int = 20
session_timeout_minutes: int = 60
max_conversation_tokens: int = 2000
```

---

## Database Schema

### Core Tables

#### Products Table
```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    sku TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    category TEXT NOT NULL,
    stock_count INTEGER DEFAULT 0,
    weight_kg DECIMAL(5,2),
    dimensions TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Orders Table
```sql
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    order_number TEXT UNIQUE NOT NULL,
    customer_name TEXT NOT NULL,
    customer_email TEXT,
    delivery_address TEXT NOT NULL,
    order_status TEXT DEFAULT 'pending',
    total_amount DECIMAL(10,2) NOT NULL,
    items TEXT NOT NULL,
    special_instructions TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Order Items Table
```sql
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_sku TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    total_price DECIMAL(10,2) NOT NULL
);
```

#### Conversations Table
```sql
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_sequence INTEGER,
    message_type TEXT,
    content TEXT,
    metadata TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Data Models (`src/database/models.py`)
Comprehensive Pydantic models with validation:
- **ProductModel**: Product information with business logic methods
- **OrderModel**: Order management with status transitions
- **OrderItemModel**: Line item validation and pricing
- **ConversationModel**: Message handling with type safety

---

## Function Calling & Tools

### Function Tools (`src/function_calling/tools.py`)
Five specialized tools for autonomous order management:

#### 1. search_products
**Purpose**: Semantic product search with filtering
**Parameters**:
- `query`: Search term for product names/descriptions
- `category`: Filter by category (electronics, books, clothing, home_goods)
- `min_price`/`max_price`: Price range filtering
- `in_stock_only`: Only show available products
- `limit`: Maximum results (1-50)
- `sort_by`: Sorting options (price, name, relevance)

#### 2. create_order
**Purpose**: Complete order creation with validation
**Parameters**:
- `customer_name`: Full name for order processing
- `customer_email`: Email for confirmation and tracking
- `delivery_address`: Complete shipping address
- `items`: List of products with SKUs and quantities
- `special_instructions`: Delivery preferences

#### 3. check_order_status
**Purpose**: Order tracking and status information
**Parameters**:
- `order_number`: Order identifier (format: ORD-XXXXXX)
- `customer_email`: Verification email (optional)

#### 4. update_order
**Purpose**: Modify existing orders
**Parameters**:
- `order_number`: Order to update
- `customer_email`: Verification required
- `delivery_address`: New shipping address
- `items`: Updated product list
- `special_instructions`: New delivery notes

#### 5. cancel_order
**Purpose**: Order cancellation with refund processing
**Parameters**:
- `order_number`: Order to cancel
- `customer_email`: Verification required
- `reason`: Cancellation reason (optional)

### Function Executor (`src/function_calling/executor.py`)
Handles tool execution with:
- Database integration for all operations
- Comprehensive error handling
- Transaction safety and rollbacks
- Business rule validation
- Result formatting and logging

---

## Conversation Management

### Memory Architecture (`src/conversation/`)
Three-layer conversation management:

#### 1. ConversationMemory (`memory.py`)
- Sliding window of last 20 messages
- Session-based storage with unique IDs
- Automatic cleanup of expired sessions
- Message sequence tracking

#### 2. OrderInformationExtractor (`extractor.py`)
- Automatic order detail extraction from conversation
- Product mention detection and association
- Intent analysis for purchase readiness
- Context-aware entity recognition

#### 3. FunctionRouter (`router.py`)
- Intelligent tool selection based on conversation context
- Intent classification with confidence scoring
- Function calling opportunity analysis
- Cross-agent coordination suggestions

### Session Lifecycle
1. **Session Creation**: Unique ID generated for each user
2. **Message Processing**: Each message added to memory with metadata
3. **Context Extraction**: Order information and intent analysis
4. **Agent Coordination**: Context shared between agents
5. **Session Cleanup**: Automatic expiration after inactivity

---

## Quality Assessment & Evaluation

### ResponseQualityEvaluator (`src/evaluator/quality_evaluator.py`)
Automated quality assessment across multiple dimensions:

#### Evaluation Dimensions
1. **Relevance** (40% weight): How well answer addresses user question
2. **Completeness** (30% weight): Comprehensive coverage of all aspects
3. **Accuracy** (30% weight): Information consistency with context

#### Evaluation Process
1. **LLM-Based Evaluation**: Uses GPT-3.5-turbo for objective assessment
2. **Structured Scoring**: 1-10 scale for each dimension
3. **Weighted Overall**: Combined score with dimension weights
4. **Improvement Recommendations**: Specific suggestions for scores < 7

#### Quality Metrics
- **Overall Score**: Weighted average (1-10 scale)
- **Dimension Scores**: Individual assessment per dimension
- **Reasoning**: Detailed explanation for scores
- **Recommendations**: Actionable improvement suggestions

---

## Observability & Monitoring

### Langfuse Integration (`src/utils/langfuse_client.py`)
Comprehensive tracing and monitoring:

#### Trace Types
- **Agent Tracing**: Individual agent execution with performance metrics
- **Function Calling**: Tool selection, execution, and results
- **RAG Pipeline**: Document retrieval and embedding generation
- **Quality Evaluation**: Assessment scores and recommendations
- **Error Tracking**: Comprehensive error logging with context

#### Performance Metrics
- **Response Times**: Agent execution and total processing time
- **Token Usage**: LLM consumption and cost estimation
- **Confidence Scores**: Agent routing and function calling confidence
- **Success Rates**: Function calling and order completion rates
- **Error Rates**: Categorized error tracking and analysis

#### Monitoring Features
- **Real-time Tracing**: Live observation of system performance
- **Health Checks**: Component status monitoring
- **Performance Analytics**: Trend analysis and optimization insights
- **Error Debugging**: Detailed error context and resolution guidance

---

## Data Sources & Embeddings

### Product Catalog Structure
32+ products organized in 4 categories:

#### Electronics (9 products)
- MacBook Pro 14-inch M3 (ELC-MBP14-256)
- iPhone 15 128GB (ELC-IP15-128)
- PlayStation 5 1TB (ELC-PS5-1TB)
- iPad, Samsung TV, Camera, Headphones, Watch, Monitor

#### Books (8 products)
- Atomic Habits, Clean Code, Sapiens, Psychology of Money
- Design Thinking, Dune Novel, Startup Ecosystem, AI Revolution

#### Clothing (8 products)
- Premium Hoodie, Denim Jacket, Organic T-Shirt, Summer Dress
- Cashmere Cardigan, 512 Jeans, Chino Pants, Comfort Sneakers

#### Home Goods (7 products)
- Robot Vacuum, Air Fryer, Coffee Maker, Desk Lamp
- Ceramic Plant, Bedding Set, Bluetooth Speaker

### Product Data Format
Each product stored as markdown with structured metadata:
```markdown
# Product Name
**SKU:** UNIQUE-IDENTIFIER
**Price:** $XXX.XX
**Category:** category_name
**Stock:** X units available

## Description
Product description and features...

## Search Terms
Relevant search terms for discovery...
```

### Vector Embeddings
- **Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **Storage**: FAISS with cosine similarity
- **Persistence**: Department-specific indices with caching
- **Updates**: Dynamic document addition and index rebuilding

### Database Integration
Real-time product information enhancement:
- **Stock Levels**: Live inventory from database
- **Pricing**: Current prices and availability
- **Product Details**: Specifications and descriptions
- **Order History**: Popular products and recommendations

---

## Dependencies

### Core Framework (`requirements.txt`)
```python
# LLM & AI Framework
langchain==1.1.3
langchain-community==0.4.1
langchain-core==1.2.0
langchain-openai==1.1.3
openai==2.11.0

# Vector Storage & Embeddings
faiss-cpu==1.13.1
sentence-transformers==5.2.0
numpy==2.3.5

# Database & ORM
sqlalchemy==2.0.45
alembic==1.17.2

# Configuration & Settings
pydantic==2.12.5
pydantic-settings==2.12.0
python-dotenv==1.2.1

# Observability & Monitoring
langfuse==3.10.6

# Testing & Utilities
pytest==9.0.2
pytest-asyncio==1.3.0
requests==2.32.5
httpx==0.28.1
urllib3==1.26.20
tqdm==4.67.1
tiktoken==0.12.0
python-dateutil==2.9.0.post0
```

### External Services
- **OpenRouter**: Primary LLM provider for chat completions
- **OpenAI**: Function calling and evaluation models
- **Langfuse**: Observability and monitoring platform

---

## Usage Examples

### Interactive Mode
```bash
# Start interactive chatbot
python -m src.main interactive

# Sample conversation
You: I'm looking for a laptop under $2000
🤖 Assistant (ProductRAGAgent): I found the MacBook Pro 14-inch M3 for $1999.99!
- M3 chip with 8-core CPU and 10-core GPU
- 14.2-inch Liquid Retina XDR display
- 16GB RAM, 256GB SSD
- 23 units in stock

You: Perfect! I'll take one
🤖 Assistant (OrderProcessingAgent): Processing your order...
📦 Order Created! Order Number: ORD-25010801
```

### Single Query Processing
```bash
# Product search
python -m src.main query --query "Find Bluetooth headphones under $100"

# Order status check
python -m src.main query --query "Check order status ORD-123456" --user-id "john@example.com"

# Order creation
python -m src.main query --query "I want to buy 2 PlayStation 5 consoles"
```

### System Management
```bash
# Initialize system
python -m src.main init

# Seed product database
python -m src.main seed

# Health check
python -m src.main status

# Force rebuild indices
python -m src.main init --force-rebuild
```

### Advanced Features
- **Multi-turn Conversations**: Context maintained across extended interactions
- **Order Extraction**: Automatic order detail detection from conversation
- **Sales Completion**: Seamless transition from browsing to purchase
- **Error Recovery**: Graceful handling of all failure scenarios
- **Performance Monitoring**: Real-time system health and performance metrics

---

## Conclusion

The M4 Delivery Chatbot represents a sophisticated transformation of the M3 multi-department system into a production-ready conversational commerce platform. Key strengths include:

### Technical Excellence
- **Modern AI Architecture**: RAG + Function Calling + Conversation Memory
- **Production-Ready**: Complete database persistence, validation, and monitoring
- **Scalable Design**: Modular components supporting easy extension
- **Comprehensive Testing**: End-to-end validation of all scenarios

### Business Value
- **Sales Conversion**: Intelligent product discovery and seamless ordering
- **Customer Experience**: Natural language interface with contextual understanding
- **Operational Efficiency**: Automated order processing with minimal human intervention
- **Data-Driven Insights**: Comprehensive analytics and performance monitoring

### Innovation Features
- **Dual-Agent Coordination**: Specialized agents with intelligent routing
- **Real-time Integration**: Live inventory and pricing accuracy
- **Quality Assurance**: Automated response evaluation and improvement
- **Observability**: Complete workflow tracing and performance optimization

This system demonstrates advanced AI engineering patterns and represents a complete solution for intelligent conversational commerce in production environments.
