document.addEventListener('DOMContentLoaded', () => {
    const initBtn = document.getElementById('init-btn');
    const submitBtn = document.getElementById('submit-btn');
    const corpusSize = document.getElementById('corpus-size');
    const corpusVal = document.getElementById('corpus-val');
    const initStatus = document.getElementById('init-status');
    const connDot = document.getElementById('conn-dot');
    const connText = document.getElementById('conn-text');

    const ragOutput = document.getElementById('rag-output');
    const baseOutput = document.getElementById('base-output');
    const sourcesOutput = document.getElementById('sources-output');

    const latVal = document.getElementById('latency-val');
    const confVal = document.getElementById('conf-val');
    const groundVal = document.getElementById('ground-val');
    const riskVal = document.getElementById('risk-val');

    let currentReference = '';
    let isInitialized = false;

    function buildHeaders() {
        const headers = { 'Content-Type': 'application/json' };
        const token = localStorage.getItem('MEDRAG_API_TOKEN');
        if (token) headers['Authorization'] = `Bearer ${token}`;
        return headers;
    }

    function setStatus(online, text) {
        connDot.className = online ? 'dot online' : 'dot offline';
        connText.textContent = text;
    }

    function renderError(target, text) {
        target.innerHTML = `<span class="placeholder" style="color:var(--danger)">${text}</span>`;
    }

    corpusSize.addEventListener('input', (e) => {
        corpusVal.textContent = e.target.value;
    });

    document.querySelectorAll('.example-list li').forEach(li => {
        li.addEventListener('click', () => {
            document.getElementById('query-input').value = li.getAttribute('data-q');
            currentReference = li.getAttribute('data-ref');
            initStatus.textContent = 'Example loaded. Ready to submit.';
            initStatus.style.color = 'var(--text-muted)';
        });
    });

    // Check health on load
    fetch('/api/v1/health/ready')
        .then(r => r.ok ? setStatus(true, 'Ready') : setStatus(false, 'Initializing'))
        .catch(() => setStatus(false, 'Offline'));

    initBtn.addEventListener('click', async () => {
        initBtn.disabled = true;
        initBtn.textContent = 'Initializing...';
        initStatus.textContent = 'Loading corpus and indexing...';
        initStatus.style.color = 'var(--text-muted)';

        try {
            const size = parseInt(corpusSize.value, 10);
            const res = await fetch('/api/v1/init', {
                method: 'POST',
                headers: buildHeaders(),
                body: JSON.stringify({ corpus_size: size })
            });
            const data = await res.json();

            if (res.ok && data.status === 'success') {
                isInitialized = true;
                initStatus.textContent = data.message;
                initStatus.style.color = 'var(--success)';
                setStatus(true, data.mode === 'full_medrag' ? 'Full MedRAG Online' : 'Fallback Mode Online');
            } else {
                isInitialized = false;
                initStatus.textContent = 'Error: ' + (data.message || 'Initialization failed');
                initStatus.style.color = 'var(--danger)';
                setStatus(false, 'Initialization failed');
            }
        } catch (err) {
            isInitialized = false;
            initStatus.textContent = 'Network error during initialization.';
            initStatus.style.color = 'var(--danger)';
            setStatus(false, 'Network error');
        } finally {
            initBtn.disabled = false;
            initBtn.textContent = 'Initialize';
        }
    });

    submitBtn.addEventListener('click', async () => {
        const query = document.getElementById('query-input').value.trim();
        if (!query) return alert('Please enter a question first.');
        if (!isInitialized) return alert('Please initialize the system before submitting a query.');

        submitBtn.disabled = true;
        submitBtn.textContent = 'Processing...';

        ragOutput.innerHTML = '<span class="placeholder">Retrieving sources and generating response...</span>';
        baseOutput.innerHTML = '<span class="placeholder">Generating baseline...</span>';

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
                    latVal.textContent = data.metrics.latency || '—';
                    confVal.textContent = data.metrics.retrieval_confidence !== undefined ? data.metrics.retrieval_confidence : '—';
                    groundVal.textContent = data.metrics.citation_grounding !== undefined ? data.metrics.citation_grounding : '—';
                    riskVal.textContent = data.metrics.hallucination_risk !== undefined ? data.metrics.hallucination_risk : '—';
                }

                sourcesOutput.innerHTML = '';
                if (data.sources && data.sources.length > 0) {
                    data.sources.forEach((s, i) => {
                        const meta = s.source ? `[${s.source}]` : '';
                        const score = s.score !== undefined ? `Score: ${s.score}` : '';
                        sourcesOutput.innerHTML += `
                            <div class="source-item" style="animation-delay: ${i * 0.05}s">
                                <div class="source-title">
                                    <span>Source ${s.rank || i + 1}: ${s.title}</span>
                                    <span class="source-meta">${meta} ${score}</span>
                                </div>
                                <div class="source-text">${s.text}</div>
                            </div>
                        `;
                    });
                } else {
                    sourcesOutput.innerHTML = '<span class="placeholder">No sources retrieved.</span>';
                }
            } else {
                const msg = data.message || 'Failed to process query.';
                renderError(ragOutput, msg);
                renderError(baseOutput, msg);
            }
        } catch (err) {
            renderError(ragOutput, 'Network error connecting to backend.');
            renderError(baseOutput, 'Network error connecting to backend.');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Submit Query';
            currentReference = '';
        }
    });
});
