import logging
import time
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


class GroqGenerator:
    def __init__(self, api_key, max_retries: int = 2):
        self.api_key = api_key
        try:
            parsed_retries = int(max_retries)
        except (TypeError, ValueError):
            parsed_retries = 2
        self.max_retries = max(0, parsed_retries)
        if not api_key:
            logger.warning("No API key provided for Groq Generator. Falling back to non-LLM response mode.")

        self.llm = (
            ChatGroq(
                api_key=api_key,
                model_name="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens=500,
            )
            if api_key
            else None
        )

        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert AI medical assistant. Answer the user's question based EXCLUSIVELY on the provided clinical context.

Context Evidence:
{context}

Patient/User Question: {question}

Instructions for generating your answer:
1. Synthesize the context explicitly. DO NOT hallucinate external facts.
2. Structure your response cleanly with bullet points if applicable.
3. If the context is insufficient, state "Context insufficient to answer".
4. Strive for high accuracy and precision in medical terminology.

Answer:""",
        )

    def _invoke_with_retry(self, messages: List, default_message: str) -> str:
        if not self.llm:
            return default_message

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                return response.content.strip()
            except Exception as exc:
                last_error = exc
                logger.warning("LLM invocation failed (attempt %s/%s): %s", attempt + 1, self.max_retries + 1, exc)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 3))

        logger.error("LLM invocation exhausted retries: %s", last_error)
        return default_message

    def generate(self, question, retrieved_docs):
        if not self.api_key or not self.llm:
            if not retrieved_docs:
                return "Context insufficient to answer"
            summary = " ".join([doc.get("text", "")[:220] for doc in retrieved_docs[:2]]).strip()
            return summary or "Context insufficient to answer"

        context = "\n\n".join([f"Document {i + 1}: {doc['text']}" for i, doc in enumerate(retrieved_docs)])
        prompt_text = self.rag_prompt.format(context=context, question=question)

        messages = [
            SystemMessage(content="You are a meticulous medical assistant focusing on accurate RAG retrieval extraction."),
            HumanMessage(content=prompt_text),
        ]
        return self._invoke_with_retry(messages, default_message="Context insufficient to answer")

    def generate_no_context(self, question):
        if not self.api_key or not self.llm:
            return "Model unavailable: baseline generation requires API credentials."

        messages = [
            SystemMessage(content="You are a medical assistant. Answer to the best of your general knowledge."),
            HumanMessage(content=question),
        ]
        return self._invoke_with_retry(messages, default_message="Unable to generate baseline answer at this time.")
