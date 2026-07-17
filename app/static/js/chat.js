document.addEventListener('DOMContentLoaded', () => {
    const chatForm      = document.getElementById('chatForm');
    const questionInput = document.getElementById('questionInput');
    const chatMessages  = document.getElementById('chatMessages');
    const sendBtn       = document.getElementById('sendBtn');
    const historyList   = document.getElementById('historyList');

    const userMsgTpl  = document.getElementById('userMsgTpl');
    const aiMsgTpl    = document.getElementById('aiMsgTpl');
    const loadingTpl  = document.getElementById('loadingTpl');

    const DOC_ID = window.DOC_ID;

    // ── Load past Q&A history into sidebar ────────────────────────────────────
    async function loadHistory() {
        try {
            const res  = await fetch(`/api/v1/documents/${DOC_ID}/history`);
            if (!res.ok) return;
            const data = await res.json();
            const msgs = (data.history || []).filter(m => m.role === 'user');
            if (msgs.length === 0) return;

            historyList.innerHTML = '';
            msgs.forEach((m, i) => {
                const el = document.createElement('div');
                el.className = 'px-2 py-1.5 rounded-lg hover:bg-brand-purple-light text-gray-700 cursor-pointer truncate text-xs leading-snug';
                el.title     = m.content;
                el.textContent = `${i + 1}. ${m.content}`;
                el.onclick = () => { questionInput.value = m.content; questionInput.focus(); };
                historyList.appendChild(el);
            });
        } catch (e) {
            console.warn('Could not load history:', e);
        }
    }

    loadHistory();

    // ── Enter to submit (Shift+Enter = newline) ────────────────────────────────
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // ── Submit ─────────────────────────────────────────────────────────────────
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const question = questionInput.value.trim();
        if (!question) return;

        // Render user bubble
        const userNode = userMsgTpl.content.cloneNode(true);
        userNode.querySelector('.msg-text').textContent = question;
        chatMessages.appendChild(userNode);

        questionInput.value = '';
        setInputEnabled(false);
        scrollToBottom();

        // Show loading spinner briefly while embedding + rerank runs
        const loadingNode = loadingTpl.content.cloneNode(true);
        chatMessages.appendChild(loadingNode);
        scrollToBottom();

        const startMs = performance.now();

        // Build the live AI bubble (hidden until first token arrives)
        const aiNode   = aiMsgTpl.content.cloneNode(true);
        const textEl   = aiNode.querySelector('.msg-text');
        const metaEl   = aiNode.querySelector('.msg-meta');
        // We need a stable reference after appendChild, so wrap in a container
        const wrapper  = document.createElement('div');
        wrapper.appendChild(aiNode);
        // Don't add to DOM yet — we'll add it on first token

        let streamedText    = '';
        let metaRendered    = false;
        let liveEl          = null;   // the actual .msg-text in the DOM
        let liveMetaEl      = null;
        let addedToDOM      = false;

        try {
            const res = await fetch(`/api/v1/documents/${DOC_ID}/query/stream`, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ question }),
            });

            if (!res.ok) {
                // Non-200 means our pipeline failed before streaming started
                removeLoading();
                const errData = await res.json().catch(() => ({}));
                renderError(errData.detail || `Server error ${res.status}`);
                return;
            }

            const reader  = res.body.getReader();
            const decoder = new TextDecoder();
            let   buffer  = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // keep incomplete last line

                for (const line of lines) {
                    if (!line.startsWith('data:')) continue;
                    const raw = line.slice(5).trim();
                    if (!raw) continue;

                    let evt;
                    try { evt = JSON.parse(raw); } catch { continue; }

                    if (evt.type === 'token') {
                        // First token: swap spinner for live bubble
                        if (!addedToDOM) {
                            removeLoading();
                            chatMessages.appendChild(wrapper);
                            liveEl     = wrapper.querySelector('.msg-text');
                            liveMetaEl = wrapper.querySelector('.msg-meta');
                            addedToDOM = true;
                        }
                        streamedText += evt.text;
                        // Render incrementally with citation highlighting
                        liveEl.innerHTML = formatAnswer(streamedText);
                        scrollToBottom();

                    } else if (evt.type === 'meta' && !metaRendered) {
                        metaRendered = true;
                        const latencyMs = Math.round(performance.now() - startMs);
                        renderMeta(liveMetaEl || metaEl, evt, latencyMs, streamedText);
                        window._lastCitations = evt.citations;

                    } else if (evt.type === 'error') {
                        removeLoading();
                        if (!addedToDOM) renderError(evt.detail || 'An error occurred.');
                        else if (liveEl) liveEl.textContent += `\n\n⚠️ ${evt.detail}`;

                    } else if (evt.type === 'done') {
                        // Stream finished
                        if (!addedToDOM) {
                            // Edge case: stream ended with no tokens (cache hit handled differently)
                            removeLoading();
                        }
                    }
                }
            }

            // Flush remaining buffer
            if (buffer.startsWith('data:')) {
                const raw = buffer.slice(5).trim();
                try {
                    const evt = JSON.parse(raw);
                    if (evt.type === 'token' && liveEl) {
                        streamedText += evt.text;
                        liveEl.innerHTML = formatAnswer(streamedText);
                    }
                } catch { /* ignore */ }
            }

            appendHistoryItem(question);

        } catch (err) {
            removeLoading();
            renderError('Network error: ' + err.message);
        } finally {
            setInputEnabled(true);
            questionInput.focus();
            scrollToBottom();
        }
    });

    // ── Format streamed answer with citation highlights ────────────────────────
    function formatAnswer(text) {
        return escapeHtml(text)
            .replace(/\n/g, '<br>')
            .replace(/\[Page (\d+)\]/g,
                '<span class="inline-flex items-center bg-brand-purple text-white text-xs px-1.5 py-0.5 rounded font-medium mx-0.5 cursor-pointer hover:bg-brand-purple-dark" onclick="showCitationAlert($1, window._lastCitations)">[Page $1]</span>'
            );
    }

    // ── Render meta badges (latency, confidence, citations) ───────────────────
    function renderMeta(metaEl, evt, latencyMs, fullText) {
        if (!metaEl) return;
        const tags = [];

        if (evt.cached) {
            tags.push(badge('Cached ⚡', 'bg-blue-100 text-blue-800'));
        } else {
            tags.push(badge(`${latencyMs}ms`, 'bg-gray-100 text-gray-600'));
        }

        if (evt.confidence !== undefined) {
            const score  = Number(evt.confidence).toFixed(2);
            const isLow  = evt.not_found === true;
            const color  = isLow ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700';
            tags.push(badge(`Confidence: ${score}`, color));
        }

        if (!evt.not_found && evt.citations && evt.citations.length > 0) {
            const pages  = [...new Set(evt.citations.map(c => c.page))].sort((a, b) => a - b);
            const citTag = document.createElement('span');
            citTag.className = 'text-brand-purple text-xs cursor-pointer hover:underline flex items-center gap-1';
            citTag.innerHTML = `📄 Pages cited: ${pages.join(', ')}`;
            citTag.onclick   = () => { window._lastCitations = evt.citations; showCitationsModal(evt.citations); };
            tags.push(citTag);
        }

        tags.forEach(t => {
            if (typeof t === 'string') metaEl.insertAdjacentHTML('beforeend', t);
            else metaEl.appendChild(t);
        });
    }

    // ── Render error bubble ────────────────────────────────────────────────────
    function renderError(msg) {
        const node   = aiMsgTpl.content.cloneNode(true);
        const textEl = node.querySelector('.msg-text');
        textEl.classList.add('bg-red-50', 'border-red-200', 'text-red-700');
        textEl.classList.remove('bg-white', 'text-gray-700', 'border-gray-100');
        textEl.textContent = '❌ ' + msg;
        chatMessages.appendChild(node);
    }

    // ── Citations modal ────────────────────────────────────────────────────────
    window.showCitationsModal = function(citations) {
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4';
        modal.onclick = (e) => { if (e.target === modal) modal.remove(); };

        let citHtml = '';
        citations.forEach(c => {
            citHtml += `
            <div class="border-l-4 border-brand-purple pl-4 py-2 mb-3">
                <div class="text-xs font-semibold text-brand-purple mb-1">Page ${c.page} — Chunk ${c.chunk}</div>
                <p class="text-sm text-gray-700 leading-relaxed">${escapeHtml(c.snippet)}</p>
            </div>`;
        });

        modal.innerHTML = `
        <div class="bg-white rounded-2xl shadow-xl max-w-lg w-full max-h-[80vh] overflow-hidden flex flex-col">
            <div class="px-6 py-4 bg-brand-purple text-white flex justify-between items-center flex-shrink-0">
                <h3 class="font-semibold">Source Citations</h3>
                <button onclick="this.closest('.fixed').remove()" class="text-white/80 hover:text-white text-xl leading-none">×</button>
            </div>
            <div class="p-6 overflow-y-auto">${citHtml}</div>
        </div>`;
        document.body.appendChild(modal);
    };

    window.showCitationAlert = function(page, citations) {
        if (!citations) return;
        const matches = citations.filter(c => c.page === page);
        if (matches.length === 0) { alert(`Page ${page} — no snippet available.`); return; }
        showCitationsModal(matches);
    };

    // ── Helpers ────────────────────────────────────────────────────────────────
    function badge(text, classes) {
        return `<span class="${classes} px-2 py-0.5 rounded-full">${text}</span>`;
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function setInputEnabled(enabled) {
        questionInput.disabled = !enabled;
        sendBtn.disabled       = !enabled;
        sendBtn.classList.toggle('opacity-50', !enabled);
    }

    function removeLoading() {
        document.getElementById('loadingIndicator')?.remove();
    }

    function scrollToBottom() {
        chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
    }

    function appendHistoryItem(question) {
        const existing = historyList.querySelectorAll('div').length;
        if (historyList.querySelector('.text-gray-400')) historyList.innerHTML = '';
        const el = document.createElement('div');
        el.className   = 'px-2 py-1.5 rounded-lg hover:bg-brand-purple-light text-gray-700 cursor-pointer truncate text-xs leading-snug';
        el.title       = question;
        el.textContent = `${existing + 1}. ${question}`;
        el.onclick     = () => { questionInput.value = question; questionInput.focus(); };
        historyList.appendChild(el);
    }
});
