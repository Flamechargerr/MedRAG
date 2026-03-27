import logging
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class GroqGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        if not api_key:
            logger.warning("No API key provided for Groq Generator. It will fail on execution.")
            
        self.llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=500
        ) if api_key else None
        
        # Advanced Prompt Engineering to ground model in structured data and improve RAG accuracy
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

Answer:"""
        )

    def generate(self, question, retrieved_docs):
        if not self.api_key or not self.llm:
            return "Error: API key is missing."
            
        context = "\n\n".join([f"Document {i+1}: {doc['text']}" for i, doc in enumerate(retrieved_docs)])
        prompt_text = self.rag_prompt.format(context=context, question=question)
        
        try:
            messages = [
                SystemMessage(content="You are a meticulous medical assistant focusing on accurate RAG retrieval extraction."),
                HumanMessage(content=prompt_text)
            ]
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"Error: {str(e)}"

    def generate_no_context(self, question):
        if not self.api_key or not self.llm:
            return "Error: API key is missing."
            
        try:
            messages = [
                SystemMessage(content="You are a medical assistant. Answer to the best of your general knowledge."),
                HumanMessage(content=question)
            ]
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Baseline API error: {e}")
            return f"Error: {str(e)}"
