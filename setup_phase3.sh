#!/bin/bash
# MedBot Phase 3 - Quick Setup Script

echo "=================================================="
echo "  MedBot Phase 3 - Environment Setup"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Check if running in Colab
if [ -d "/content" ]; then
    echo "✅ Google Colab detected"
    IN_COLAB=true
else
    echo "ℹ️  Running locally"
    IN_COLAB=false
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take 5-10 minutes..."

if [ "$IN_COLAB" = true ]; then
    # Colab-specific installation
    pip install -q torch transformers datasets chromadb openai sentence-transformers \
        faiss-cpu numpy pandas matplotlib seaborn scikit-learn rouge-score tqdm accelerate tiktoken
else
    # Local installation
    pip install -r requirements.txt
fi

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; print('✅ GPU Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Download required models (optional pre-download)
echo ""
echo "Pre-downloading embedding model (optional)..."
python3 -c "
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')
try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print('✅ Embedding model downloaded')
except Exception as e:
    print(f'⚠️  Download skipped: {e}')
"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p chroma_db
mkdir -p models
mkdir -p results

echo ""
echo "=================================================="
echo "  ✅ Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Launch Jupyter: jupyter notebook MedBot_Phase3.ipynb"
echo "   OR open in Google Colab"
echo ""
echo "2. Run all cells sequentially"
echo ""
echo "3. Results will be saved in current directory"
echo ""
echo "For help, see PHASE3_README.md"
echo "=================================================="
