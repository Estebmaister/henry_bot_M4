# 🔍 Observability and Tracing Manual

## Table of Contents
1. [Overview](#overview)
2. [Langfuse Integration](#langfuse-integration)
3. [Configuration](#configuration)
4. [Trace Structure](#trace-structure)
5. [Monitoring Dashboard](#monitoring-dashboard)
6. [Debugging with Traces](#debugging-with-traces)
7. [Performance Metrics](#performance-metrics)
8. [Alerting and Monitoring](#alerting-and-monitoring)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

## Overview

The Multi-Agent Intelligent Routing System includes comprehensive observability through **Langfuse** integration, providing complete workflow tracing, performance monitoring, and debugging capabilities.

### What Gets Traced?

- **Query Intent Classification**: Department routing decisions with confidence scores
- **Agent Execution**: Performance metrics for each specialized agent
- **RAG Retrieval**: Document similarity scores and retrieval times
- **Quality Evaluation**: Automated response scoring and recommendations
- **Error Handling**: Detailed error tracking and context
- **System Performance**: End-to-end processing times and bottlenecks

## Langfuse Integration

### Architecture
```
Query → Multi-Agent System → Langfuse Traces → Dashboard
                     ↓
               All Components Observed
```

### Key Components

1. **Trace Level**: Complete query processing workflow
2. **Span Level**: Individual component execution (classification, retrieval, generation)
3. **Event Level**: Specific occurrences (classification results, errors)
4. **Generation Level**: LLM calls and quality evaluations

## Configuration

### 1. Langfuse Setup

1. **Create Langfuse Account**:
   - Visit [https://cloud.langfuse.com](https://cloud.langfuse.com)
   - Sign up for free tier (up to 50,000 traces/month)

2. **Get API Keys**:
   ```bash
   # In your Langfuse dashboard
   Settings → API Keys → Create New Key
   ```

3. **Configure Environment**:
   ```bash
   # Edit .env file
   LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxx
   LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxx
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

### 2. Automatic Configuration

The system automatically initializes Langfuse when credentials are provided:

```python
from src.utils import langfuse_client

# Automatic initialization
if langfuse_client.enabled:
    print("Langfuse tracing enabled")
else:
    print("Langfuse disabled (missing credentials)")
```

## Trace Structure

### Main Trace: "multi_agent_query_processing"

**Input**: User query and metadata
**Output**: Final answer and performance metrics

### Nested Observations:

#### 1. Intent Classification
```json
{
  "name": "intent_classification",
  "type": "event",
  "input": "What benefits am I entitled to?",
  "output": "hr",
  "metadata": {
    "confidence": 0.94,
    "all_scores": {"hr": 0.94, "tech": 0.12, "finance": 0.08},
    "threshold": 0.7,
    "classification_method": "semantic_similarity"
  }
}
```

#### 2. RAG Retrieval
```json
{
  "name": "rag_retrieval",
  "type": "span",
  "input": "What benefits am I entitled to?",
  "output": "Retrieved 3 documents",
  "metadata": {
    "num_documents": 3,
    "similarity_scores": [0.94, 0.87, 0.82],
    "average_similarity": 0.88,
    "retrieval_time_seconds": 0.234,
    "embedding_model": "all-MiniLM-L6-v2"
  }
}
```

#### 3. Agent Execution
```json
{
  "name": "HR Assistant_execution",
  "type": "span",
  "input": "What benefits am I entitled to?",
  "output": "As a new employee, you are entitled to...",
  "metadata": {
    "agent_name": "HR Assistant",
    "agent_type": "rag_hr",
    "execution_time_seconds": 1.456,
    "num_retrieved_docs": 3,
    "retrieval_success": true
  }
}
```

#### 4. Quality Evaluation
```json
{
  "name": "quality_evaluation",
  "type": "generation",
  "input": "What benefits am I entitled to?",
  "output": "As a new employee, you are entitled to...",
  "metadata": {
    "context_length": 2847,
    "answer_length": 456,
    "quality_scores": {
      "relevance": 9.0,
      "completeness": 8.0,
      "accuracy": 8.5
    },
    "overall_score": 8.5,
    "evaluator_model": "gpt-3.5-turbo"
  }
}
```

#### 5. Error Events (when applicable)
```json
{
  "name": "error",
  "type": "event",
  "input": "...",
  "output": "Agent processing failed: connection timeout",
  "metadata": {
    "error_type": "agent_processing_error",
    "timestamp": "2025-11-18T15:30:45.123Z",
    "agent_name": "HR Assistant",
    "department": "hr"
  }
}
```

## Monitoring Dashboard

### 1. Access Your Dashboard

1. **Login to Langfuse**: [https://cloud.langfuse.com](https://cloud.langfuse.com)
2. **Select Your Project**: Henry bot M3 - Multi-Agent System
3. **View Real-time Traces**: Monitor queries as they process

### 2. Key Dashboard Features

#### **Traces Overview**
- Real-time trace visualization
- Filter by time range, user, or department
- Search specific queries
- Export traces for analysis

#### **Performance Metrics**
- Average processing time by department
- Classification confidence distribution
- Quality score trends
- Error rates and types

#### **Component Analysis**
- Individual agent performance
- RAG retrieval efficiency
- Quality evaluation consistency

### 3. Essential Views

#### **Trace Details View**
```
Query: "What benefits am I entitled to?"
├── Intent Classification (0.05s) ✅
│   └── Department: hr (94% confidence)
├── RAG Retrieval (0.23s) ✅
│   └── 3 documents retrieved (avg similarity: 0.88)
├── HR Assistant Execution (1.45s) ✅
│   └── Answer generated
├── Quality Evaluation (0.67s) ✅
│   └── Score: 8.5/10
└── Total Time: 2.4s
```

#### **Performance Analytics**
- Processing time distribution
- Classification accuracy trends
- Quality score breakdown by dimension
- Error frequency and types

## Debugging with Traces

### 1. Common Debugging Scenarios

#### **Misclassification Issues**

**Problem**: Query routed to wrong department

**Debug Steps**:
1. **Search the query** in Langfuse dashboard
2. **Examine classification metadata**:
   ```json
   "classification": {
     "predicted": "tech",
     "confidence": 0.65,
     "scores": {"hr": 0.62, "tech": 0.65, "finance": 0.08}
   }
   ```
3. **Analysis**: Low confidence score indicates borderline classification
4. **Solution**: Adjust confidence threshold or improve training examples

#### **Poor Quality Responses**

**Problem**: Low quality scores for certain queries

**Debug Steps**:
1. **Filter traces by quality score** (< 7.0)
2. **Examine quality evaluation metadata**:
   ```json
   "quality_scores": {
     "relevance": 6.0,
     "completeness": 5.0,
     "accuracy": 7.0
   },
   "recommendations": [
     "Answer lacks specific benefit details",
     "Missing enrollment process information"
   ]
   ```
3. **Review retrieved documents**: Check if relevant context was found
4. **Update documentation**: Add missing information to source documents

#### **Performance Issues**

**Problem**: Slow processing times

**Debug Steps**:
1. **Sort traces by processing time** (descending)
2. **Identify bottlenecks**:
   ```
   RAG Retrieval: 2.3s (should be < 0.5s)
   Agent Execution: 4.1s (should be < 2s)
   ```
3. **Optimization opportunities**:
   - Increase FAISS index size
   - Optimize document chunking
   - Use faster LLM models

### 2. Advanced Debugging Techniques

#### **Trace Correlation**
```python
# Add custom metadata for correlation
trace = langfuse_client.create_trace(
    name="multi_agent_query_processing",
    input=query,
    user_id=user_id,
    metadata={
        "session_id": "sess_12345",
        "query_type": "benefits_inquiry",
        "user_tier": "new_employee"
    }
)
```

#### **Custom Event Logging**
```python
# Add custom debugging events
langfuse_client.log_agent_execution(
    trace=trace,
    agent_name="custom_debug",
    agent_type="debugging",
    input_data="debug_checkpoint",
    output_data="retrieval_completed",
    execution_time=checkpoint_time,
    metadata={"checkpoint": "post_retrieval"}
)
```

## Performance Metrics

### 1. Key Performance Indicators (KPIs)

#### **System Performance**
- **Average Processing Time**: Target < 3 seconds
- **95th Percentile Latency**: Target < 5 seconds
- **Error Rate**: Target < 2%
- **Classification Accuracy**: Target > 90%

#### **Quality Metrics**
- **Average Quality Score**: Target > 7.5/10
- **Relevance Score**: Target > 8.0/10
- **Completeness Score**: Target > 7.0/10
- **Accuracy Score**: Target > 8.0/10

#### **Department Performance**
- **HR Queries**: Avg processing time, quality score
- **Tech Queries**: Avg processing time, quality score
- **Finance Queries**: Avg processing time, quality score

### 2. Monitoring in Langfuse

#### **Dashboards**
Create custom dashboards for:

1. **Real-time Monitoring**
   ```sql
   -- Example metric query
   SELECT
     AVG(metadata.processing_time) as avg_time,
     COUNT(*) as query_count,
     DATE_TRUNC('hour', timestamp) as hour
   FROM traces
   WHERE name = 'multi_agent_query_processing'
     AND timestamp >= NOW() - INTERVAL '24 hours'
   GROUP BY hour
   ```

2. **Quality Trends**
   ```sql
   SELECT
     AVG(metadata.quality_evaluation.overall_score) as avg_quality,
     metadata.department,
     DATE_TRUNC('day', timestamp) as day
   FROM traces
   WHERE metadata.quality_evaluation IS NOT NULL
   GROUP BY department, day
   ```

#### **Alerts**
Set up alerts for:

- **High Latency**: Processing time > 5 seconds
- **Low Quality**: Quality score < 6.0
- **High Error Rate**: Error rate > 5%
- **Misclassification**: Confidence < 0.6 with multiple departments close

## Alerting and Monitoring

### 1. Setting Up Alerts

#### **In Langfuse Dashboard**

1. **Navigate to Settings → Alerts**
2. **Create New Alert**:
   - Name: "High Processing Time Alert"
   - Condition: `processing_time > 5`
   - Channel: Email, Slack, or Webhook
   - Threshold: 5 consecutive occurrences

3. **Quality Alert**:
   - Name: "Low Quality Score Alert"
   - Condition: `quality_score < 6.0`
   - Channel: Email
   - Threshold: 3 occurrences in 1 hour

## Advanced Features

### 1. Custom Trace Metadata

Enhance traces with business context:

```python
# In your query processing
response = await system.process_query(
    query,
    user_id=user_id,
    custom_metadata={
        "user_department": "engineering",
        "employee_level": "senior",
        "query_category": "benefits",
        "session_id": session_id
    }
)
```

### 2. A/B Testing with Traces

Compare different approaches:

```python
# Method A: Current approach
trace_a = langfuse_client.create_trace(
    name="method_a_query_processing",
    input=query,
    metadata={"method": "baseline"}
)

# Method B: Experimental approach
trace_b = langfuse_client.create_trace(
    name="method_b_query_processing",
    input=query,
    metadata={"method": "experimental_v2"}
)
```

## Troubleshooting

### 1. Common Issues and Solutions

#### **Langfuse Not Working**

**Problem**: No traces appearing in dashboard

**Symptoms**:
- `Langfuse disabled: missing credentials` message
- No error messages, but no traces

**Solutions**:
1. **Check Environment Variables**:
   ```bash
   echo $LANGFUSE_SECRET_KEY
   echo $LANGFUSE_PUBLIC_KEY
   ```

2. **Verify API Keys**:
   - Ensure keys are valid and active
   - Check for extra spaces or special characters

3. **Test Connection**:
   ```python
   from src.utils import langfuse_client

   if langfuse_client.enabled:
       trace = langfuse_client.create_trace(
           name="test_trace",
           input="test"
       )
       langfuse_client.flush()
       print("Test trace created successfully")
   ```

#### **Missing Trace Data**

**Problem**: Traces appear but have incomplete data

**Solutions**:
1. **Check for Exception Handling**:
   ```python
   # Ensure traces are completed even on errors
   try:
       # Your processing logic
       response = await process_query(query, trace)
   except Exception as e:
       # Log error to trace
       langfuse_client.log_error(trace, str(e), "processing_error")
       raise
   finally:
       # Always flush traces
       langfuse_client.flush()
   ```

2. **Verify Metadata Serialization**:
   ```python
   # Ensure metadata is JSON serializable
   metadata = {
       "processing_time": float(time_taken),  # Convert to JSON types
       "document_count": int(len(docs))
   }
   ```

#### **High Memory Usage**

**Problem**: System consuming too much memory with tracing

**Solutions**:
1. **Adjust Trace Sampling**:
   ```python
   # Sample only 10% of traces in production
   if random.random() < 0.1:  # 10% sampling
       trace = langfuse_client.create_trace(...)
   ```

2. **Limit Metadata Size**:
   ```python
   # Truncate large content in metadata
   def safe_metadata(data):
       if isinstance(data, str) and len(data) > 1000:
           return data[:1000] + "..."
       return data
   ```

### 2. Performance Optimization

#### **Reduce Trace Overhead**
```python
# Optimize trace creation for high-volume systems
class OptimizedTracing:
    def __init__(self, sample_rate=0.1):
        self.sample_rate = sample_rate
        self.buffer = []
        self.buffer_size = 100

    def add_trace_data(self, trace_data):
        if random.random() < self.sample_rate:
            self.buffer.append(trace_data)
            if len(self.buffer) >= self.buffer_size:
                self.flush_buffer()

    def flush_buffer(self):
        # Batch flush traces
        for data in self.buffer:
            langfuse_client.create_trace(**data)
        langfuse_client.flush()
        self.buffer.clear()
```

### 3. Best Practices

#### **DO's** ✅
- Always include user context in traces
- Use consistent metadata structure
- Set up alerts for key metrics
- Regular review of trace patterns
- Export and analyze performance data

#### **DON'Ts** ❌
- Don't include sensitive PII in traces
- Don't create traces with extremely large metadata
- Don't ignore trace failures silently
- Don't forget to flush traces on shutdown
- Don't use tracing for security logging (use proper security tools instead)

### 4. 📊 Tracing - Quick Reference Cheatsheet **Essential Langfuse Queries**

```sql
-- 1. Top slow queries today
SELECT input, metadata.processing_time, metadata.department
FROM traces
WHERE DATE(timestamp) = CURRENT_DATE
ORDER BY metadata.processing_time DESC
LIMIT 10;

-- 2. Quality score distribution
SELECT
  CASE
    WHEN metadata.quality_evaluation.overall_score >= 8 THEN 'Good (8-10)'
    WHEN metadata.quality_evaluation.overall_score >= 6 THEN 'Fair (6-8)'
    ELSE 'Poor (<6)'
  END as quality_category,
  COUNT(*) as count
FROM traces
WHERE metadata.quality_evaluation IS NOT NULL
GROUP BY quality_category;

-- 3. Department performance comparison
SELECT
  metadata.department,
  COUNT(*) as queries,
  AVG(metadata.processing_time) as avg_time,
  AVG(metadata.quality_evaluation.overall_score) as avg_quality
FROM traces
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY metadata.department;

-- 4. Error analysis
SELECT
  observations.metadata.error_type,
  COUNT(*) as error_count,
  metadata.department
FROM traces, observations
WHERE observations.type = 'error'
GROUP BY observations.metadata.error_type, metadata.department;
```

---

**This manual provides comprehensive guidance for leveraging the full observability capabilities of the Hnery bot M3 - Multi-Agent Intelligent Routing System. Regular monitoring and analysis of traces will help maintain system performance and identify opportunities for improvement.**