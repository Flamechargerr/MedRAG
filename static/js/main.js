document.addEventListener('DOMContentLoaded', () => {
    const initBtn = document.getElementById('init-btn');
    const submitBtn = document.getElementById('submit-btn');
    const corpusSize = document.getElementById('corpus-size');
    const corpusVal = document.getElementById('corpus-val');
    const initStatus = document.getElementById('init-status');
    const connectionDot = document.getElementById('connection-dot');
    const connectionText = document.getElementById('connection-text');
    
    // Output fields
    const ragOutput = document.getElementById('rag-output');
    const baseOutput = document.getElementById('base-output');
    const sourcesOutput = document.getElementById('sources-output');
    
    // Metric fields
    const latVal = document.getElementById('latency-val');
    const accVal = document.getElementById('acc-val');
    const ragVal = document.getElementById('rag-val');
    const baseVal = document.getElementById('base-val');
    
    let currentReference = "";

    // Sync slider
    corpusSize.addEventListener('input', (e) => {
        corpusVal.textContent = e.target.value;
    });

    // Handle Examples
    document.querySelectorAll('#examples-list li').forEach(li => {
        li.addEventListener('click', () => {
            document.getElementById('query-input').value = li.getAttribute('data-q');
            currentReference = li.getAttribute('data-ref');
            initStatus.textContent = "Example scenario loaded. Ready.";
            initStatus.style.color = "var(--text-muted)";
        });
    });

    // Initialize System
    initBtn.addEventListener('click', async () => {
        initBtn.disabled = true;
        initBtn.textContent = 'Initializing...';
        initStatus.textContent = 'Indexing medical documents into FAISS...';
        initStatus.style.color = 'var(--text-muted)';
        
        try {
            const res = await fetch('/api/init', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ corpus_size: parseInt(corpusSize.value) })
            });
            const data = await res.json();
            
            if(data.status === 'success') {
                initStatus.textContent = data.message;
                initStatus.style.color = 'var(--success)';
                connectionDot.className = 'dot online';
                connectionText.textContent = 'System Online & Indexed';
            } else {
                initStatus.textContent = 'Error: ' + data.message;
                initStatus.style.color = 'var(--danger)';
            }
        } catch(err) {
            initStatus.textContent = 'Network error during initialization.';
            initStatus.style.color = 'var(--danger)';
        } finally {
            initBtn.disabled = false;
            initBtn.textContent = 'Initialize Engine';
        }
    });

    // Submit Query
    submitBtn.addEventListener('click', async () => {
        const query = document.getElementById('query-input').value.trim();
        if(!query) return alert('Enter a patient clinical presentation or medical query first.');
        
        submitBtn.disabled = true;
        submitBtn.textContent = 'Processing RAG Pipeline...';
        
        ragOutput.innerHTML = '<span class="placeholder-text">Analyzing evidence & synthesizing response...</span>';
        baseOutput.innerHTML = '<span class="placeholder-text">Generating direct inference...</span>';
        
        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, reference: currentReference })
            });
            const data = await res.json();
            
            if(data.status === 'success') {
                ragOutput.textContent = data.answer;
                baseOutput.textContent = data.baseline_answer;
                
                // Set metrics
                if(data.metrics) {
                    latVal.textContent = data.metrics.Latency || '0.00s';
                    accVal.textContent = data.metrics.Accuracy_Improvement || 'N/A';
                    ragVal.textContent = data.metrics.RAG_ROUGE_L || 'N/A';
                    baseVal.textContent = data.metrics.Baseline_ROUGE_L || 'N/A';
                }
                
                // Set sources
                sourcesOutput.innerHTML = '';
                if(data.sources && data.sources.length > 0) {
                    data.sources.forEach((s, i) => {
                        sourcesOutput.innerHTML += `
                            <div class="source-item" style="animation-delay: ${i * 0.1}s">
                                <div class="source-title">Source ${i+1}: ${s.title}</div>
                                <div class="source-text">${s.text}</div>
                            </div>
                        `;
                    });
                } else {
                    sourcesOutput.innerHTML = '<span class="placeholder-text">No highly relevant sources found.</span>';
                }
            } else {
                alert('Backend Error: ' + data.message);
                ragOutput.innerHTML = '<span class="placeholder-text" style="color:var(--danger)">Failed to process query.</span>';
                baseOutput.innerHTML = '<span class="placeholder-text" style="color:var(--danger)">Failed to process query.</span>';
            }
        } catch(err) {
            alert('Network Error');
            ragOutput.innerHTML = '<span class="placeholder-text" style="color:var(--danger)">Network error connecting to backend.</span>';
            baseOutput.innerHTML = '<span class="placeholder-text" style="color:var(--danger)">Network error connecting to backend.</span>';
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Run Diagnostic RAG';
            currentReference = ""; // Reset reference after processing
        }
    });
});
