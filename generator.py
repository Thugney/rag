import logging
import time
from typing import Any, Dict, Iterator, List, Optional

from llm_factory import LLMFactory, LLMProviderSettings
from vector_store import RetrievalResult


logger = logging.getLogger(__name__)


class OpenAICompatibleGenerator:
    """
    Response generation through an OpenAI-compatible chat-completions endpoint.

    DeepSeek remains the default runtime provider, but the generator is now
    configured through provider settings so we can swap in OpenAI-compatible
    endpoints without changing the application layer.
    """

    PROMPT_VERSION = "1.2.0"

    def __init__(self, settings: LLMProviderSettings, verbosity: str = "normal"):
        self.settings = settings
        self.provider = settings.provider
        self.model_name = settings.model_name
        self.temperature = settings.temperature
        self.max_tokens = settings.max_tokens
        self.verbosity = verbosity
        self._client = None

        if settings.configured:
            logger.info(
                "Initialized %s generator for model '%s' with prompt version %s",
                settings.display_name,
                self.model_name,
                self.PROMPT_VERSION,
            )
        else:
            logger.warning(
                "Initialized %s generator without credentials. Chat generation will stay unavailable until %s is set.",
                settings.display_name,
                settings.api_key_env,
            )

    @property
    def client(self):
        if self._client is None:
            self._client = LLMFactory.create_client(self.settings)
        return self._client

    def validate_connection(self) -> Dict[str, Any]:
        return LLMFactory.validate_connection(self.settings)

    def generate_response(self, query: str, context_chunks: List[RetrievalResult]) -> Iterator[str]:
        """Generate a streamed response using the configured provider."""
        start_time = time.time()
        if not context_chunks:
            logger.warning("No context chunks provided for query response")
            yield "I could not find any relevant information in the uploaded documents to answer your question."
            return

        validation = self.validate_connection()
        if not validation["valid"]:
            logger.warning("Skipping generation because provider validation failed: %s", validation["message"])
            yield f"Generation is unavailable: {validation['message']}"
            return

        context_text = self._format_context(context_chunks)
        query_type = self._classify_query(query)
        few_shot_examples = self._get_few_shot_examples(query_type)

        system_prompt = self._build_system_prompt()
        user_content = self._build_user_content(query, context_text, few_shot_examples)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            logger.info(
                "Generating response with provider=%s model=%s query_type=%s",
                self.provider,
                self.model_name,
                query_type,
            )
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            response_content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content
                    response_content += content_piece
                    yield content_piece

            latency = time.time() - start_time
            logger.info("Response generated in %.2f seconds", latency)
            self._log_response_metrics(query, response_content, latency, query_type)
        except Exception as exc:
            logger.error("Error during %s API call: %s", self.settings.display_name, exc)
            yield f"Sorry, I encountered an error while communicating with {self.settings.display_name}."

    def _build_system_prompt(self) -> str:
        verbosity_instruction = {
            "low": "Provide brief responses with minimal detail.",
            "normal": "Provide balanced responses with necessary detail.",
            "high": "Provide detailed responses with comprehensive explanations.",
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
        return f"""CONTEXT FROM DOCUMENTS:
{context_text}

FEW-SHOT EXAMPLES FOR GUIDANCE:
{few_shot_examples}

USER QUERY:
{query}"""

    def _classify_query(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ["how", "process", "steps", "method"]):
            return "procedural"
        if any(word in query_lower for word in ["what", "define", "explain", "meaning"]):
            return "explanatory"
        if any(word in query_lower for word in ["when", "date", "time", "history"]):
            return "temporal"
        if any(word in query_lower for word in ["who", "person", "people", "individual"]):
            return "personal"
        return "general"

    def _get_few_shot_examples(self, query_type: str) -> str:
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
A: Yes, this solution is scalable and can handle up to 10,000 concurrent users without performance degradation. [Source: technical_specs.docx]""",
        }
        return examples.get(query_type, examples["general"])

    def _format_context(self, context_chunks: List[RetrievalResult]) -> str:
        formatted_context = []
        seen_contents: Dict[str, List[tuple[str, str, float]]] = {}

        for index, result in enumerate(context_chunks):
            source = result.chunk.metadata.get("filename", "Unknown Source")
            content = result.chunk.content
            chunk_id = f"{source}_{index}"

            if content in seen_contents:
                seen_contents[content].append((chunk_id, source, result.score))
            else:
                seen_contents[content] = [(chunk_id, source, result.score)]

            formatted_context.append(
                f"Source ID: {chunk_id}\n"
                f"Source File: {source} (Relevance Score: {result.score:.2f})\n"
                f"Content: {content}"
            )

        conflict_note = ""
        for _, sources in seen_contents.items():
            if len(sources) > 1:
                source_info = "; ".join([f"{source_name} (Score: {score:.2f})" for _, source_name, score in sources])
                conflict_note += (
                    f"\nNote: Potential conflict detected - identical content found in multiple sources: {source_info}"
                )

        return "\n\n---\n\n".join(formatted_context) + conflict_note

    def _log_response_metrics(self, query: str, response: str, latency: float, query_type: str) -> None:
        metrics = {
            "query_length": len(query),
            "response_length": len(response),
            "latency_seconds": latency,
            "query_type": query_type,
            "model_used": self.model_name,
            "provider": self.provider,
            "prompt_version": self.PROMPT_VERSION,
        }
        logger.info("Response metrics: %s", metrics)


class DeepSeekGenerator(OpenAICompatibleGenerator):
    """
    Backward-compatible alias for the previous generator entrypoint.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        verbosity: str = "normal",
        max_tokens: int = 4096,
        base_url: Optional[str] = "https://api.deepseek.com/v1",
        api_key_env: str = "DEEPSEEK_API_KEY",
    ):
        settings = LLMProviderSettings(
            provider="deepseek",
            display_name="DeepSeek",
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key_env=api_key_env,
            base_url=base_url,
        )
        super().__init__(settings=settings, verbosity=verbosity)


class GeneratorFactory:
    @staticmethod
    def from_config(config: Any, verbosity: str = "normal") -> OpenAICompatibleGenerator:
        settings = LLMFactory.from_config(config)
        return OpenAICompatibleGenerator(settings=settings, verbosity=verbosity)
