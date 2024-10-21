"""Legacy metrics with ROUGE-based evaluation."""
from rouge_score import rouge_scorer
import time

class RAGEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    
    def compute_rouge_scores(self, generated, reference):
        scores = self.scorer.score(str(reference), str(generated))
        return {'rouge1': scores['rouge1'].fmeasure, 'rouge2': scores['rouge2'].fmeasure, 'rougeL': scores['rougeL'].fmeasure}
    
    def detect_hallucination(self, answer, docs):
        if not docs or not answer: return 1.0
        answer_tokens = set(str(answer).lower().split())
        doc_tokens = set()
        for d in docs: doc_tokens.update(str(d.get('text','')).lower().split())
        if not answer_tokens: return 1.0
        return 1.0 - len(answer_tokens & doc_tokens) / len(answer_tokens)
    
    def measure_response_time(self, func, *args, **kwargs):
        start = time.time(); result = func(*args, **kwargs); return result, time.time() - start
