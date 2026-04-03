import os
import logging
import time
from typing import List, Iterator, Dict, Any
from openai import OpenAI
from vector_store import RetrievalResult

logger = logging.getLogger(__name__)

class DeepSeekGenerator:
    """
    DeepSeek-based response generation using OpenAI-compatible API with advanced prompt engineering.
    """
    # Prompt versioning
    PROMPT_VERSION = "1.1.0"
    
    def __init__(self, model_name: str, temperature: float, verbosity: str = "normal"):
        self.model_name = model_name
        self.temperature = temperature
        self.verbosity = verbosity
        
        # Initialize OpenAI client with DeepSeek API
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            error_msg = "DEEPSEEK_API_KEY not found in .env file."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        logger.info(f"Initialized DeepSeekGenerator with prompt version {self.PROMPT_VERSION}")
    
    def _verify_model(self):
        """Verify that the API key is valid by listing available models."""
        try:
            self.client.models.list()
            logger.info(f"DeepSeek API client initialized successfully for model '{self.model_name}'.")
        except Exception as e:
            logger.error(f"Failed to connect to DeepSeek API. Check your API key. Error: {e}")
            raise

    def generate_response(self, query: str, context_chunks: List[RetrievalResult]) -> Iterator[str]:
        """Generate a streamed response using the DeepSeek Chat Completions API with advanced prompting."""
        start_time = time.time()
        if not context_chunks:
            logger.warning("No context chunks provided for query response")
            yield "I could not find any relevant information in the uploaded documents to answer your question."
            return

        context_text = self._format_context(context_chunks)
        query_type = self._classify_query(query)
        few_shot_examples = self._get_few_shot_examples(query_type)
        
        # Base system prompt with versioning
        system_prompt = self._build_system_prompt()
        
        # User message with structured context and query
        user_content = self._build_user_content(query, context_text, few_shot_examples)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            logger.info(f"Generating response for query type: {query_type}")
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                temperature=self.temperature
            )
            response_content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content
                    response_content += content_piece
                    yield content_piece
            latency = time.time() - start_time
            logger.info(f"Response generated in {latency:.2f} seconds")
            # Log metrics for monitoring
            self._log_response_metrics(query, response_content, latency, query_type)
        except Exception as e:
            logger.error(f"Error during DeepSeek API call: {e}")
            yield "Sorry, I encountered an error while communicating with the DeepSeek API."

    def _build_system_prompt(self) -> str:
        """Builds the system prompt with versioning and structured instructions."""
        verbosity_instruction = {
            "low": "Provide brief responses with minimal detail.",
            "normal": "Provide balanced responses with necessary detail.",
            "high": "Provide detailed responses with comprehensive explanations."
        }.get(self.verbosity, "Provide balanced responses with necessary detail.")
        
        return f"""You are a helpful and factual AI assistant (Prompt Version: {self.PROMPT_VERSION}).
Your purpose is to assist users by providing accurate information based ONLY on the provided context from uploaded documents.
Follow these guidelines strictly:
- {verbosity_instruction}
- Structure your response in markdown format with clear headings, bullet points, and numbered lists for readability.
- When sources conflict, explicitly acknowledge the conflict, present all perspectives, and indicate which perspective (if any) appears more authoritative based on recency or source credibility.
- Always cite sources using the format [Source: filename] after relevant statements.
- If uncertain about an answer or if the context is incomplete, clearly signal uncertainty with phrases like "Based on limited information..." or "I'm not certain, but...".
- If the context does not contain the answer, state: "The information is not available in the provided documents."
- Do not invent information or make assumptions beyond what is provided in the context."""

    def _build_user_content(self, query: str, context_text: str, few_shot_examples: str) -> str:
        """Builds structured user content with context, query, and few-shot examples."""
        return f"""CONTEXT FROM DOCUMENTS:
{context_text}

FEW-SHOT EXAMPLES FOR GUIDANCE:
{few_shot_examples}

USER QUERY:
{query}"""

    def _classify_query(self, query: str) -> str:
        """Classifies the query type for dynamic few-shot example selection."""
        query_lower = query.lower()
        if any(word in query_lower for word in ["how", "process", "steps", "method"]):
            return "procedural"
        elif any(word in query_lower for word in ["what", "define", "explain", "meaning"]):
            return "explanatory"
        elif any(word in query_lower for word in ["when", "date", "time", "history"]):
            return "temporal"
        elif any(word in query_lower for word in ["who", "person", "people", "individual"]):
            return "personal"
        else:
            return "general"

    def _get_few_shot_examples(self, query_type: str) -> str:
        """Returns few-shot examples based on query type."""
        examples = {
            "procedural": """Example 1:
Q: How do I reset the system?
A: To reset the system, follow these steps:
  - Power off the device.
  - Wait for 30 seconds.
  - Power on the device while holding the reset button for 10 seconds.
  [Source: manual.pdf]

Example 2:
Q: What are the steps to install the software?
A: Installing the software involves:
  1. Downloading the installer from the official website.
  2. Running the installer and following the on-screen instructions.
  3. Restarting your computer after installation.
  [Source: installation_guide.docx]""",
            
            "explanatory": """Example 1:
Q: What is cloud computing?
A: Cloud computing is a model for delivering on-demand computing resources over the internet, including storage, processing power, and software. It allows users to access technology services without managing physical servers. [Source: tech_glossary.pdf]

Example 2:
Q: Explain the concept of machine learning.
A: Machine learning is a subset of artificial intelligence where systems learn from data to improve performance on specific tasks without being explicitly programmed. It involves training models on datasets to make predictions or decisions. [Source: ai_textbook.pdf]""",
            
            "temporal": """Example 1:
Q: When was the company founded?
A: The company was founded in 1995. [Source: company_history.pdf]

Example 2:
Q: What happened during the 2020 incident?
A: During the 2020 incident, a major system outage occurred due to a cyberattack, affecting operations for 48 hours. [Source: incident_report.docx]""",
            
            "personal": """Example 1:
Q: Who is the CEO of the company?
A: The CEO of the company is Jane Smith. [Source: annual_report.pdf]

Example 2:
Q: Who invented this technology?
A: This technology was invented by Dr. John Doe in 2010. [Source: patent_document.docx]""",
            
            "general": """Example 1:
Q: What are the benefits of this product?
A: The benefits of this product include:
  - Increased efficiency by 30%.
  - Reduced costs by up to 25%.
  - Improved user satisfaction ratings.
  [Source: product_brochure.pdf]

Example 2:
Q: Is this solution scalable?
A: Yes, this solution is scalable and can handle up to 10,000 concurrent users without performance degradation. [Source: technical_specs.docx]"""
        }
        return examples.get(query_type, examples["general"])

    def _format_context(self, context_chunks: List[RetrievalResult]) -> str:
        """Formats the context chunks for the prompt with conflict detection."""
        formatted_context = []
        seen_contents: Dict[str, List[tuple]] = {}
        
        for i, res in enumerate(context_chunks):
            source = res.chunk.metadata.get('filename', 'Unknown Source')
            content = res.chunk.content
            chunk_id = f"{source}_{i}"
            
            if content in seen_contents:
                seen_contents[content].append((chunk_id, source, res.score))
            else:
                seen_contents[content] = [(chunk_id, source, res.score)]
                
            formatted_context.append(f"Source ID: {chunk_id}\nSource File: {source} (Relevance Score: {res.score:.2f})\nContent: {content}")
        
        # Check for conflicts
        conflict_note = ""
        for content, sources in seen_contents.items():
            if len(sources) > 1:
                source_info = "; ".join([f"{src[1]} (Score: {src[2]:.2f})" for src in sources])
                conflict_note += f"\nNote: Potential conflict detected - identical content found in multiple sources: {source_info}"
        
        return "\n\n---\n\n".join(formatted_context) + conflict_note

    def _log_response_metrics(self, query: str, response: str, latency: float, query_type: str) -> None:
        """Logs metrics for monitoring response quality and performance."""
        metrics = {
            "query_length": len(query),
            "response_length": len(response),
            "latency_seconds": latency,
            "query_type": query_type,
            "model_used": self.model_name,
            "prompt_version": self.PROMPT_VERSION
        }
        logger.info(f"Response metrics: {metrics}")
