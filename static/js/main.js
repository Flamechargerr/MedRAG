document.addEventListener('DOMContentLoaded', () => {
    const initBtn = document.getElementById('init-btn');
    const submitBtn = document.getElementById('submit-btn');
    const corpusSize = document.getElementById('corpus-size');
    const corpusVal = document.getElementById('corpus-val');
    const initStatus = document.getElementById('init-status');
    const connectionDot = document.getElementById('connection-dot');
    const connectionText = document.getElementById('connection-text');

    const ragOutput = document.getElementById('rag-output');
    const baseOutput = document.getElementById('base-output');
    const sourcesOutput = document.getElementById('sources-output');

    const latVal = document.getElementById('latency-val');
    const accVal = document.getElementById('acc-val');
    const ragVal = document.getElementById('rag-val');
    const baseVal = document.getElementById('base-val');

    let currentReference = "";
    let isInitialized = false;

    function buildHeaders() {
        const headers = { 'Content-Type': 'application/json' };
        const token = localStorage.getItem('MEDRAG_API_TOKEN');
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        return headers;
    }

    function setSystemStatus(online, text) {
        connectionDot.className = online ? 'dot online' : 'dot offline';
        connectionText.textContent = text;
    }

    function renderError(target, text) {
        target.innerHTML = `<span class="placeholder-text" style="color:var(--danger)">${text}</span>`;
    }

    corpusSize.addEventListener('input', (e) => {
        corpusVal.textContent = e.target.value;
    });

    document.querySelectorAll('#examples-list li').forEach(li => {
        li.addEventListener('click', () => {
            document.getElementById('query-input').value = li.getAttribute('data-q');
            currentReference = li.getAttribute('data-ref');
            initStatus.textContent = "Example scenario loaded. Ready.";
            initStatus.style.color = "var(--text-muted)";
        });
    });

    initBtn.addEventListener('click', async () => {
        initBtn.disabled = true;
        initBtn.textContent = 'Initializing...';
        initStatus.textContent = 'Indexing medical documents into FAISS...';
        initStatus.style.color = 'var(--text-muted)';

        try {
            const parsedCorpus = parseInt(corpusSize.value, 10);
            const maxCorpus = parseInt(corpusSize.max, 10);
            if (
                Number.isNaN(parsedCorpus) ||
                parsedCorpus <= 0 ||
                (!Number.isNaN(maxCorpus) && parsedCorpus > maxCorpus)
            ) {
                throw new Error('Invalid corpus size selected.');
            }

            const res = await fetch('/api/v1/init', {
                method: 'POST',
                headers: buildHeaders(),
                body: JSON.stringify({ corpus_size: parsedCorpus })
            });
            const data = await res.json();

            if (res.ok && data.status === 'success') {
                isInitialized = true;
                initStatus.textContent = data.message;
                initStatus.style.color = 'var(--success)';
                setSystemStatus(true, 'System Online & Indexed');
            } else {
                isInitialized = false;
                initStatus.textContent = 'Error: ' + (data.message || 'Initialization failed');
                initStatus.style.color = 'var(--danger)';
                setSystemStatus(false, 'Initialization failed');
            }
        } catch (err) {
            isInitialized = false;
            initStatus.textContent = 'Network error during initialization.';
            initStatus.style.color = 'var(--danger)';
            setSystemStatus(false, 'Network error');
        } finally {
            initBtn.disabled = false;
            initBtn.textContent = 'Initialize Engine';
        }
    });

    submitBtn.addEventListener('click', async () => {
        const query = document.getElementById('query-input').value.trim();
        if (!query) {
            return alert('Enter a patient clinical presentation or medical query first.');
        }
        if (!isInitialized) {
            return alert('Please initialize the system before submitting a query.');
        }

        submitBtn.disabled = true;
        submitBtn.textContent = 'Processing RAG Pipeline...';

        ragOutput.innerHTML = '<span class="placeholder-text">Analyzing evidence & synthesizing response...</span>';
        baseOutput.innerHTML = '<span class="placeholder-text">Generating direct inference...</span>';

        try {
            const res = await fetch('/api/v1/chat', {
                method: 'POST',
                headers: buildHeaders(),
                body: JSON.stringify({ query, reference: currentReference })
            });
            const data = await res.json();

            if (res.ok && data.status === 'success') {
                ragOutput.textContent = data.answer;
                baseOutput.textContent = data.baseline_answer;

                if (data.metrics) {
                    latVal.textContent = data.metrics.Latency !== undefined ? data.metrics.Latency : '0.00s';
                    accVal.textContent = data.metrics.Accuracy_Improvement !== undefined ? data.metrics.Accuracy_Improvement : 'N/A';
                    ragVal.textContent = data.metrics.RAG_ROUGE_L !== undefined ? data.metrics.RAG_ROUGE_L : 'N/A';
                    baseVal.textContent = data.metrics.Baseline_ROUGE_L !== undefined ? data.metrics.Baseline_ROUGE_L : 'N/A';
                }

                sourcesOutput.innerHTML = '';
                if (data.sources && data.sources.length > 0) {
                    data.sources.forEach((s, i) => {
                        sourcesOutput.innerHTML += `
                            <div class="source-item" style="animation-delay: ${i * 0.1}s">
                                <div class="source-title">Source ${i + 1}: ${s.title}</div>
                                <div class="source-text">${s.text}</div>
                            </div>
                        `;
                    });
                } else {
                    sourcesOutput.innerHTML = '<span class="placeholder-text">No highly relevant sources found.</span>';
                }
            } else {
                const errorMessage = data.message || 'Failed to process query.';
                alert('Backend Error: ' + errorMessage);
                renderError(ragOutput, errorMessage);
                renderError(baseOutput, errorMessage);
            }
        } catch (err) {
            alert('Network Error');
            renderError(ragOutput, 'Network error connecting to backend.');
            renderError(baseOutput, 'Network error connecting to backend.');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Run Diagnostic RAG';
            currentReference = "";
        }
    });
});
