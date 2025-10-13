# MedBot Phase 3 - Implementation Guide

## 📋 Overview

This Phase 3 implementation includes:
- **Baseline LSTM Retriever Model** for embedding-based search
- **ChatGPT RAG** (GPT-3.5-Turbo) via Emergent LLM Key  
- **Llama-2 Medical RAG** (medical-tuned model)
- **ChromaDB** for efficient vector storage and retrieval
- **Comprehensive Evaluation**: Retrieval F1, ROUGE-1/2/L, hallucination detection, response time
- **Visualizations**: Comparative plots and analysis

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Upload `MedBot_Phase3.ipynb` to Google Colab
2. Set Runtime to GPU: `Runtime > Change runtime type > GPU`
3. Run all cells sequentially
4. Results will be saved automatically

### Option 2: Local Jupyter

```bash
# Install Jupyter
pip install jupyter

# Launch notebook
jupyter notebook MedBot_Phase3.ipynb
```

### Option 3: JupyterLab

```bash
# Install JupyterLab
pip install jupyterlab

# Launch
jupyter lab MedBot_Phase3.ipynb
```

## 📦 Dependencies

All dependencies are installed automatically in the notebook. Key packages:

```
torch>=2.1.0
transformers==4.44.2
chromadb>=0.4.0
openai==0.28.0
sentence-transformers==2.2.2
datasets==2.16.1
rouge-score>=0.1.2
numpy, pandas, matplotlib, seaborn, scikit-learn
```

See `requirements.txt` for complete list.

## 🔑 API Keys Configuration

### Emergent LLM Key (for ChatGPT)
The notebook is pre-configured with the Emergent LLM Key: `sk-emergent-56016CcDc780e503a4`

This provides access to:
- OpenAI GPT-3.5-Turbo
- GPT-4 (if needed)
- Other supported models

### Hugging Face Token (Optional)
For gated models like Llama-2, set environment variable:

```python
import os
os.environ['HUGGINGFACE_API_KEY'] = 'your_hf_token_here'
```

Or export in terminal:
```bash
export HUGGINGFACE_API_KEY='your_token'
```

## 📊 What Gets Generated

### Models
1. **lstm_retriever_model.pt** - Trained baseline LSTM retriever
2. **chroma_db/** - ChromaDB vector index directory

### Results
1. **phase3_results.csv** - Detailed results for each question
2. **phase3_aggregate_stats.csv** - Aggregated statistics per system
3. **phase3_summary.txt** - Human-readable summary report

### Visualizations
1. **rouge_scores_comparison.png** - ROUGE-1/2/L comparison
2. **hallucination_rate_comparison.png** - Hallucination rates
3. **response_time_breakdown.png** - Time analysis
4. **comprehensive_radar_chart.png** - Multi-metric comparison

### Package
**medbot_phase3_results.zip** - All results bundled together

## 🎯 Evaluation Metrics

### 1. Retrieval F1 Score
Measures precision and recall of document retrieval against ground truth.

### 2. ROUGE Scores
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap  
- **ROUGE-L**: Longest common subsequence

### 3. Hallucination Rate
Token overlap analysis between generated answers and retrieved documents. Lower is better.

### 4. Response Time
Total time = Retrieval time + Generation time

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              MedBot Phase 3 Pipeline            │
├─────────────────────────────────────────────────┤
│                                                 │
│  Question → [Retrieval] → [Generation] → Answer│
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  Retrieval Options:                            │
│  • Baseline LSTM (trained)                     │
│  • ChromaDB (semantic search)                  │
│                                                 │
│  Generation Options:                           │
│  • ChatGPT (GPT-3.5-Turbo)                    │
│  • Llama-2 Medical (7B)                       │
│  • Baseline (template)                        │
│                                                 │
└─────────────────────────────────────────────────┘
```

## 📈 Expected Performance

Based on MedQA benchmarks:

| System | ROUGE-1 | ROUGE-L | Hallucination | Speed |
|--------|---------|---------|---------------|-------|
| Baseline LSTM | ~0.25 | ~0.20 | High (~0.7) | Fast |
| ChatGPT RAG | ~0.45 | ~0.40 | Low (~0.3) | Medium |
| Llama-2 RAG | ~0.40 | ~0.35 | Medium (~0.4) | Slow |

*Note: Actual results may vary based on dataset and configuration*

## 🔧 Customization

### Modify Configuration

In Cell 4 of the notebook:

```python
CONFIG = {
    'device': device,
    'max_length': 512,              # Token length
    'batch_size': 16,                # Batch size
    'embedding_dim': 384,            # Embedding dimension
    'lstm_hidden_dim': 256,          # LSTM hidden size
    'lstm_num_layers': 2,            # LSTM layers
    'num_epochs': 5,                 # Training epochs
    'learning_rate': 0.001,          # Learning rate
    'top_k_retrieval': 5,            # Top-K documents
    'num_eval_questions': 100,       # Evaluation size
}
```

### Change Dataset

```python
# Use different medical QA dataset
dataset = load_dataset("medmcqa", split="validation")
# or
dataset = load_dataset("pubmed_qa", "pqa_labeled", split="test")
```

### Use Different Models

```python
# For embedding
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# For generation
llama_model_name = "meta-llama/Llama-2-13b-chat-hf"  # Larger model
```

## ⚠️ Troubleshooting

### GPU Memory Issues

If you encounter CUDA OOM errors:

1. Reduce batch size: `CONFIG['batch_size'] = 8`
2. Use smaller model: `'all-MiniLM-L6-v2'` instead of larger embeddings
3. Reduce evaluation size: `CONFIG['num_eval_questions'] = 50`

### Llama-2 Loading Fails

The notebook includes fallback mechanisms. If Llama-2 doesn't load:
- Evaluation continues with Baseline + ChatGPT
- Mock responses may be used for testing

### API Rate Limits

If ChatGPT API hits rate limits:
- Add delays: `time.sleep(1)` between requests
- Reduce evaluation size
- Consider using batch processing

### Dataset Loading Issues

The notebook includes multiple fallback datasets:
1. MedQA-USMLE (primary)
2. MedMCQA (fallback)
3. Synthetic corpus (last resort)

## 📝 Implementation Details

### Baseline LSTM Architecture

```
Embedding Layer (vocab_size → 384)
    ↓
Bidirectional LSTM (384 → 512)
    ↓
Mean Pooling
    ↓
Fully Connected (512 → embedding_dim)
    ↓
L2 Normalization
```

### ChromaDB Configuration

- **Collection**: medical_corpus
- **Distance**: Cosine similarity
- **Backend**: DuckDB + Parquet
- **Persistence**: ./chroma_db/

### RAG Pipeline

1. **Query Encoding**: Embed question using SentenceTransformer
2. **Retrieval**: Top-K similar documents from ChromaDB
3. **Context Formation**: Concatenate retrieved documents
4. **Generation**: LLM generates answer with context
5. **Evaluation**: Compute metrics against reference

## 🎓 Research Notes

### Training Strategy
- **Contrastive Learning**: LSTM learns to maximize similarity with relevant documents
- **Regularization**: Dropout (0.3) + gradient clipping
- **Optimization**: Adam with learning rate 0.001

### Evaluation Considerations
- **Retrieval F1**: Challenging without explicit ground truth documents
- **ROUGE Scores**: Standard for text generation quality
- **Hallucination**: Simple token overlap (can be improved with fact-checking models)
- **Response Time**: Includes both retrieval and generation

## 🚀 Next Steps

### Immediate Improvements
1. **Better Training Data**: Use question-document pairs with labels
2. **Hybrid Retrieval**: Combine BM25 + dense retrieval
3. **Re-ranking**: Add cross-encoder for retrieved documents
4. **Advanced Hallucination**: Use NLI models for fact verification

### Production Deployment
1. **API Wrapper**: FastAPI service for inference
2. **Caching**: Redis for frequently asked questions
3. **Monitoring**: Track metrics in production
4. **A/B Testing**: Compare model versions

### Research Extensions
1. **Multi-hop RAG**: Iterative retrieval for complex questions
2. **Query Refinement**: Improve retrieval with query expansion
3. **Domain Adaptation**: Fine-tune on medical corpora
4. **Ensemble Methods**: Combine multiple retrievers/generators

## 📚 References

1. **MedRAG Paper**: [Benchmarking RAG for Medicine](https://aclanthology.org/2024.findings-acl.372/)
2. **ChromaDB**: [Documentation](https://docs.trychroma.com/)
3. **Sentence Transformers**: [Documentation](https://www.sbert.net/)
4. **ROUGE Score**: [Lin 2004](https://aclanthology.org/W04-1013/)

## 💡 Tips for Best Results

1. **Use GPU**: Colab with T4 GPU recommended (15GB VRAM)
2. **Run Sequentially**: Execute cells in order
3. **Monitor Progress**: Check tqdm progress bars
4. **Review Logs**: Check for warnings/errors
5. **Validate Results**: Inspect sample outputs before full evaluation

## 📞 Support

For questions or issues:
1. Check troubleshooting section above
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify API keys are correctly set

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Maintainer**: MedBot Phase 3 Team

✅ **Ready to run!** Open `MedBot_Phase3.ipynb` and start evaluating! 🚀
