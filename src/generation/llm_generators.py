import logging
import openai
from transformers import AutoTokenizer, pipeline
import torch

logger = logging.getLogger(__name__)

class GroqGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
        openai.api_base = "https://api.groq.com/openai/v1"
        if not api_key:
            logger.warning("No API key provided for Groq Generator. It will fail on execution.")

    def generate(self, question, retrieved_docs):
        if not self.api_key:
            return "Error: API key is missing."
            
        context = "\\n\\n".join([f"Document {i+1}: {doc['text']}" for i, doc in enumerate(retrieved_docs)])
        
        prompt = f"""You are a medical expert. Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Provide a clear, concise answer based on the context. If the context doesn't contain enough information, indicate that."""
        
        try:
            response = openai.ChatCompletion.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"ChatGPT API error: {e}")
            return f"Error: {str(e)}"


class LlamaMedicalGenerator:
    def __init__(self, hf_token, model_name="ruslanmv/Medical-Llama2-7B"):
        self.hf_token = hf_token
        self.model_name = model_name
        self.pipeline = None
        self._initialize_model()

    def _initialize_model(self):
        logger.info(f"Loading {self.model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=tokenizer,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                max_new_tokens=500,
                token=self.hf_token
            )
            logger.info("Llama model loaded successfully.")
        except Exception as e:
            logger.error(f"Llama loading error: {e}. Model will not be available.")

    def generate(self, question, retrieved_docs):
        if self.pipeline is None:
            return "[Llama model not available - using fallback response]"
        
        context = "\\n\\n".join([f"Context {i+1}: {doc['text']}" for i, doc in enumerate(retrieved_docs)])
        
        prompt = f"""### Context:
{context}

### Question:
{question}

### Answer:
"""
        
        try:
            output = self.pipeline(
                prompt,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                num_return_sequences=1,
            )
            answer = output[0]['generated_text'].split('### Answer:')[-1].strip()
            return answer
        except Exception as e:
            logger.error(f"Llama generation error: {e}")
            return f"Error: {str(e)}"


def baseline_generate(question, retrieved_docs):
    """Baseline deterministic generator: returns top doc content."""
    if not retrieved_docs:
        return "No relevant information found."
    return f"Based on the retrieved information: {retrieved_docs[0]['text'][:300]}..."
