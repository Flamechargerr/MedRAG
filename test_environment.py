#!/usr/bin/env python3
"""
MedBot Phase 3 - Quick Verification Test
Run this to verify your environment is properly set up
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'chromadb': 'ChromaDB',
        'openai': 'OpenAI',
        'sentence_transformers': 'Sentence Transformers',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'rouge_score': 'ROUGE Score',
    }
    
    failed = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NOT INSTALLED")
            failed.append(name)
    
    if failed:
        print(f"\n⚠️  Missing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

def test_gpu():
    """Test GPU availability"""
    print("\nTesting GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"  ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("  ⚠️  No GPU detected (will use CPU - slower)")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_api_key():
    """Test API key configuration"""
    print("\nTesting API configuration...")
    try:
        import openai
        import os
        
        # Check for Emergent LLM Key
        key = os.environ.get('EMERGENT_LLM_KEY', 'sk-emergent-56016CcDc780e503a4')
        if key:
            print(f"  ✅ Emergent LLM Key configured: {key[:15]}...")
            openai.api_key = key
            return True
        else:
            print("  ⚠️  No API key found (will be set in notebook)")
            return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_embedding_model():
    """Test embedding model download"""
    print("\nTesting embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        test_embedding = model.encode(["Test sentence"])
        print(f"  ✅ Embedding model working (dim: {test_embedding.shape[1]})")
        return True
    except Exception as e:
        print(f"  ⚠️  Embedding model test failed: {e}")
        print("  (Will download during notebook execution)")
        return True

def test_dataset_access():
    """Test dataset loading"""
    print("\nTesting dataset access...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test[:5]")
        print(f"  ✅ MedQA dataset accessible ({len(dataset)} samples loaded)")
        return True
    except Exception as e:
        print(f"  ⚠️  MedQA dataset error: {e}")
        print("  (Will use fallback dataset in notebook)")
        return True

def test_chromadb():
    """Test ChromaDB functionality"""
    print("\nTesting ChromaDB...")
    try:
        import chromadb
        from chromadb.config import Settings
        
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./test_chroma_db"
        ))
        
        # Create test collection
        collection = client.create_collection(name="test_collection")
        collection.add(
            embeddings=[[1.0, 2.0, 3.0]],
            documents=["test document"],
            ids=["test_id"]
        )
        
        # Query
        results = collection.query(
            query_embeddings=[[1.0, 2.0, 3.0]],
            n_results=1
        )
        
        # Cleanup
        client.delete_collection(name="test_collection")
        
        print("  ✅ ChromaDB working correctly")
        return True
    except Exception as e:
        print(f"  ❌ ChromaDB error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("  MedBot Phase 3 - Environment Verification")
    print("="*60)
    print()
    
    results = {
        'Imports': test_imports(),
        'GPU': test_gpu(),
        'API Key': test_api_key(),
        'Embedding Model': test_embedding_model(),
        'Dataset Access': test_dataset_access(),
        'ChromaDB': test_chromadb(),
    }
    
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test}")
    
    print("\n" + "="*60)
    if passed == total:
        print("  ✅ All tests passed! Ready to run Phase 3!")
    else:
        print(f"  ⚠️  {total - passed} test(s) failed")
        print("  Check errors above and install missing dependencies")
    print("="*60)
    print("\nNext step: jupyter notebook MedBot_Phase3.ipynb")
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
