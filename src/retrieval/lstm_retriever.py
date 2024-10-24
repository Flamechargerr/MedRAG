import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class LSTMRetriever(nn.Module):
    """Simple LSTM-based retriever for creating query embeddings"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim):
        super(LSTMRetriever, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), 
                                                       batch_first=True, enforce_sorted=False)
            lstm_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        pooled = torch.mean(lstm_out, dim=1)
        output = self.fc(pooled)
        return F.normalize(output, p=2, dim=1)


def train_lstm_retriever(model, train_loader, num_epochs, learning_rate, corpus_embeddings, device):
    """Train LSTM retriever with contrastive learning"""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CosineEmbeddingLoss()
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lengths = attention_mask.sum(dim=1)
            
            # Forward pass
            query_embeddings = model(input_ids, lengths)
            
            # Contrastive loss using random negative sampling
            batch_size = query_embeddings.size(0)
            random_indices = torch.randint(0, len(corpus_embeddings), (batch_size,))
            target_embeddings = torch.from_numpy(corpus_embeddings[random_indices]).float().to(device)
            labels = torch.ones(batch_size).to(device)
            
            loss = criterion(query_embeddings, target_embeddings, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    return model


def baseline_lstm_retrieve(question, tokenizer, lstm_model, corpus_embeddings, medical_corpus, device, top_k=5):
    """Retrieve relevant documents using trained LSTM model"""
    lstm_model.eval()
    with torch.no_grad():
        encoded = tokenizer(
            question,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        lengths = attention_mask.sum(dim=1)
        
        query_embedding = lstm_model(input_ids, lengths)
        query_embedding = query_embedding.cpu().numpy()
        
        similarities = np.dot(corpus_embeddings, query_embedding.T).squeeze()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        retrieved_docs = [
            {
                'text': medical_corpus[idx]['text'],
                'similarity': float(similarities[idx]),
                'metadata': {'title': medical_corpus[idx]['title']}
            }
            for idx in top_indices
        ]
    return retrieved_docs
