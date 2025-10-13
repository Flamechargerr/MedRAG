#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MedBot Phase 3 - Final Production System                  ║
║                                                                              ║
║  Developer: Anamay                                                           ║
║  Repository: https://github.com/MarcusV210/MedBot/tree/Anamay              ║
║                                                                              ║
║  3 Models:                                                                   ║
║    1. Baseline LSTM (Trained on Harrison's - 20 epochs, LR=0.3)            ║
║    2. Medical RAG Model 1 (GPT-3.5-Turbo with medical prompting)           ║
║    3. Medical RAG Model 2 (GPT-4o-mini - Advanced medical reasoning)       ║
║                                                                              ║
║  Dataset: Harrison's Principles of Internal Medicine (~15,000 pages)        ║
║  Evaluation: 25 Medical Q&A pairs from FAQ_Test.csv                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, time, warnings, re, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import List, Dict

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# RAG Components
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# Document Processing
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

print("\n" + "="*90)
print("🏥 MedBot Phase 3 - Complete Medical RAG System")
print("="*90)
print("\n📋 System Configuration:")
print("   • 3 Models: Baseline LSTM + 2 Medical RAG Models")
print("   • Dataset: Harrison's Principles (~15,000 pages)")
print("   • Training: 20 epochs with validation")
print("   • Evaluation: 25 medical Q&A pairs")
print("\n" + "="*90 + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

OPENROUTER_API_KEY = "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"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎯 Computing Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# Training Hyperparameters (OPTIMIZED)
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
OUTPUT_DIM = 768
BATCH_SIZE = 128
LEARNING_RATE = 0.3  # High LR as requested
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-4

# Data Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_PAGES_TO_PROCESS = 3000  # Process 3000 pages from 15K for balance of speed/quality

print("⚙️  Training Configuration:")
print(f"   • Epochs: {NUM_EPOCHS}")
print(f"   • Batch Size: {BATCH_SIZE}")
print(f"   • Learning Rate: {LEARNING_RATE} (High for fast convergence)")
print(f"   • Pages to Process: {MAX_PAGES_TO_PROCESS} from ~15,000")
print(f"   • Chunk Size: {CHUNK_SIZE} with {CHUNK_OVERLAP} overlap")
print("\n" + "="*90 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD MEDICAL Q&A DATASET
# ═══════════════════════════════════════════════════════════════════════════════

print("📚 STEP 1: Loading Medical Q&A Dataset")
print("-" * 90)

try:
    qa_df = pd.read_csv('FAQ_Test.csv')
    MEDICAL_QA = []
    for _, row in qa_df.iterrows():
        MEDICAL_QA.append({
            "question": row['Question'],
            "expected_answer": row['Expected_answer']
        })
    print(f"✅ Loaded {len(MEDICAL_QA)} medical Q&A pairs from FAQ_Test.csv")
    print(f"   Sample question: {MEDICAL_QA[0]['question'][:80]}...")
except Exception as e:
    print(f"❌ Error loading FAQ_Test.csv: {e}")
    exit(1)

print("\n" + "="*90 + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: LOAD AND PROCESS HARRISON'S TEXTBOOK
# ═══════════════════════════════════════════════════════════════════════════════

print("📖 STEP 2: Loading Harrison's Principles of Internal Medicine")
print("-" * 90)

def clean_medical_text(text: str) -> str:
    """Clean medical text while preserving important information"""
    # Remove page numbers and headers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*Page\s+\d+.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # Fix hyphenated words
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Normalize whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

try:
    # Find Harrison's PDF
    pdf_files = glob.glob("data/Harrison*Medicine*.pdf")
    if not pdf_files:
        raise FileNotFoundError("Harrison's PDF not found in data/ folder")
    
    pdf_path = pdf_files[0]
    print(f"📄 Found: {pdf_path}")
    
    # Load PDF
    print(f"⏳ Loading PDF pages...")
    loader = PyPDFLoader(pdf_path)
    all_pages = loader.load()
    
    total_pages = len(all_pages)
    print(f"📊 Total pages in PDF: {total_pages:,}")
    
    # Process main medical content (skip index and references)
    # Harrison's structure: ~168 pages front matter, ~1200 pages back matter
    useful_pages = all_pages[168:-1201]
    print(f"📋 Useful medical content: {len(useful_pages):,} pages")
    
    # Process subset for speed while maintaining quality
    pages_to_process = useful_pages[:MAX_PAGES_TO_PROCESS]
    print(f"🔄 Processing {len(pages_to_process):,} pages for training...")
    
    # Clean pages
    cleaned_texts = []
    for page in tqdm(pages_to_process, desc="Cleaning pages", ncols=80):
        text = clean_medical_text(page.page_content)
        if len(text) > 100:  # Keep substantial content
            cleaned_texts.append(text)
    
    print(f"✅ Cleaned {len(cleaned_texts):,} pages")
    
    # Chunk documents with overlap
    print(f"✂️  Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    medical_chunks = []
    for i, text in enumerate(tqdm(cleaned_texts, desc="Creating chunks", ncols=80)):
        chunks = text_splitter.split_text(text)
        for j, chunk in enumerate(chunks):
            if len(chunk) > 50:
                medical_chunks.append({
                    'text': chunk,
                    'source': 'Harrison\'s Principles',
                    'page': i + 168,
                    'chunk_id': f"page_{i}_chunk_{j}"
                })
    
    print(f"✅ Created {len(medical_chunks):,} medical text chunks")
    print(f"   Average chunk size: {np.mean([len(c['text']) for c in medical_chunks]):.0f} characters")
    
except Exception as e:
    print(f"❌ Error loading Harrison's: {e}")
    print("⚠️  Using sample medical data for demonstration")
    medical_chunks = [
        {'text': f"Sample medical text chunk {i}", 'source': 'Sample', 'page': i, 'chunk_id': f"sample_{i}"}
        for i in range(1000)
    ]

print("\n" + "="*90 + "\n")
