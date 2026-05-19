document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const subjectInput = document.getElementById('analysis-subject');
    const inputContainer = document.querySelector('.input-container');
    const sendBtn = document.getElementById('send-btn');
    const loader = document.getElementById('loader');
    const modeButtons = Array.from(document.querySelectorAll('.mode-btn'));

    let currentMode = 'chat';

    modeButtons.forEach((button) => {
        button.addEventListener('click', () => setMode(button.dataset.mode));
    });

    userInput.addEventListener('input', autoResize);

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    function setMode(mode) {
        currentMode = mode === 'analysis' ? 'analysis' : 'chat';
        modeButtons.forEach((button) => {
            const isActive = button.dataset.mode === currentMode;
            button.classList.toggle('active', isActive);
            button.setAttribute('aria-selected', String(isActive));
        });

        const isAnalysis = currentMode === 'analysis';
        subjectInput.classList.toggle('hidden', !isAnalysis);
        inputContainer.classList.toggle('analysis-mode', isAnalysis);
        userInput.placeholder = isAnalysis
            ? 'İncelenecek politika, taslak veya olay metnini yazın...'
            : 'Anayasa hakkında bir soru sorun...';
        userInput.setAttribute('aria-label', isAnalysis ? 'İnceleme metni' : 'Anayasa sorusu');
        sendBtn.setAttribute('aria-label', isAnalysis ? 'İncelemeyi gönder' : 'Soruyu gönder');
        autoResize();
    }

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        const request = buildRequest(text);
        appendMessage('user', request.displayText);
        userInput.value = '';
        autoResize();

        setLoading(true);
        chatBox.scrollTop = chatBox.scrollHeight;

        try {
            const response = await fetch(request.endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(request.body)
            });

            if (!response.ok) {
                throw new Error(await getErrorMessage(response));
            }

            const data = await response.json();
            appendMessage('bot', data.answer, {
                confidence: data.confidence,
                citations: data.citations || [],
                reviewNotes: data.review_notes || [],
                riskLevel: data.risk_level,
                status: data.status,
                signals: data.signals || []
            });
        } catch (error) {
            appendMessage('bot', `Hata: ${error.message} - Sistem şu anda çevrimdışı veya yükleniyor olabilir.`);
        } finally {
            setLoading(false);
        }
    }

    function buildRequest(text) {
        if (currentMode === 'analysis') {
            const subject = subjectInput.value.trim() || 'Genel anayasal inceleme';
            return {
                endpoint: '/api/v1/analyze',
                body: {
                    subject,
                    description: text,
                    mode: 'policy'
                },
                displayText: `${subject}\n\n${text}`
            };
        }

        return {
            endpoint: '/chat',
            body: { query: text },
            displayText: text
        };
    }

    async function getErrorMessage(response) {
        try {
            const data = await response.json();
            return data?.error?.message || 'Sunucu ile iletişim kurulamadı.';
        } catch (_) {
            return 'Sunucu ile iletişim kurulamadı.';
        }
    }

    function setLoading(isLoading) {
        loader.classList.toggle('hidden', !isLoading);
        sendBtn.disabled = isLoading;
        userInput.disabled = isLoading;
        subjectInput.disabled = isLoading;
    }

    function autoResize() {
        userInput.style.height = '54px';
        userInput.style.height = `${Math.min(userInput.scrollHeight, 180)}px`;
    }

    function confidenceLabel(confidence) {
        const labels = {
            verified: 'Alıntı doğrulandı',
            source_grounded: 'Kaynaklı yanıt',
            needs_review: 'Kontrol gerekli'
        };
        return labels[confidence] || labels.needs_review;
    }

    function riskLabel(level) {
        const labels = {
            low: 'Düşük risk',
            medium: 'Orta risk',
            high: 'Yüksek risk',
            critical: 'Kritik risk'
        };
        return labels[level] || labels.medium;
    }

    function statusLabel(status) {
        const labels = {
            no_clear_issue: 'Belirgin sorun yok',
            review_recommended: 'İnceleme önerilir',
            high_risk: 'Yüksek dikkat',
            needs_review: 'Kontrol gerekli'
        };
        return labels[status] || labels.needs_review;
    }

    function appendGuardrailMetadata(container, meta = {}) {
        const hasCitations = Array.isArray(meta.citations) && meta.citations.length > 0;
        const hasNotes = Array.isArray(meta.reviewNotes) && meta.reviewNotes.length > 0;
        const hasSignals = Array.isArray(meta.signals) && meta.signals.length > 0;
        if (!meta.confidence && !hasCitations && !hasNotes && !meta.riskLevel && !hasSignals) return;

        const panel = document.createElement('div');
        panel.className = 'guardrail-panel';

        if (meta.confidence) {
            const badge = document.createElement('div');
            badge.className = `confidence-badge confidence-${meta.confidence}`;
            badge.textContent = confidenceLabel(meta.confidence);
            panel.appendChild(badge);
        }

        if (meta.riskLevel || meta.status || hasSignals) {
            appendAnalysisMeta(panel, meta);
        }

        if (hasCitations) {
            const citations = document.createElement('div');
            citations.className = 'citation-list';

            meta.citations.forEach((citation) => {
                const item = document.createElement('details');
                item.className = 'citation-item';

                const summary = document.createElement('summary');
                summary.textContent = `${citation.title || 'Kaynak'} (${citation.source || 'constitution.json'})`;

                const excerpt = document.createElement('p');
                excerpt.textContent = citation.excerpt || '';

                item.appendChild(summary);
                item.appendChild(excerpt);
                citations.appendChild(item);
            });

            panel.appendChild(citations);
        }

        if (hasNotes) {
            const notes = document.createElement('ul');
            notes.className = 'review-notes';

            meta.reviewNotes.slice(0, 4).forEach((note) => {
                const item = document.createElement('li');
                item.textContent = note;
                notes.appendChild(item);
            });

            panel.appendChild(notes);
        }

        container.appendChild(panel);
    }

    function appendAnalysisMeta(panel, meta) {
        const analysisMeta = document.createElement('div');
        analysisMeta.className = 'analysis-meta';

        const line = document.createElement('div');
        line.className = 'risk-line';

        if (meta.riskLevel) {
            const risk = document.createElement('div');
            risk.className = `risk-badge risk-${meta.riskLevel}`;
            risk.textContent = riskLabel(meta.riskLevel);
            line.appendChild(risk);
        }

        if (meta.status) {
            const status = document.createElement('div');
            status.className = 'status-badge';
            status.textContent = statusLabel(meta.status);
            line.appendChild(status);
        }

        if (line.childNodes.length > 0) {
            analysisMeta.appendChild(line);
        }

        if (Array.isArray(meta.signals) && meta.signals.length > 0) {
            const signals = document.createElement('div');
            signals.className = 'signal-list';

            meta.signals.slice(0, 4).forEach((signal) => {
                const item = document.createElement('div');
                item.className = 'signal-item';

                const title = document.createElement('div');
                title.className = 'signal-title';
                title.textContent = signal.label || 'Risk sinyali';

                const details = document.createElement('div');
                const terms = (signal.matched_terms || []).slice(0, 4).join(', ');
                const hints = (signal.article_hints || []).map((hint) => `Madde ${hint}`).join(', ');
                details.textContent = [terms, hints].filter(Boolean).join(' | ');

                item.appendChild(title);
                item.appendChild(details);
                signals.appendChild(item);
            });

            analysisMeta.appendChild(signals);
        }

        panel.appendChild(analysisMeta);
    }

    function appendMessage(sender, text, meta = {}) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerText = sender === 'user' ? 'Siz' : 'AI';

        const content = document.createElement('div');
        content.className = 'msg-content';

        if (sender === 'bot') {
            content.innerHTML = window.marked ? marked.parse(text) : escapeHtml(text).replace(/\n/g, '<br>');
            appendGuardrailMetadata(content, meta);
        } else {
            content.innerText = text;
        }

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);

        chatBox.appendChild(msgDiv);

        setTimeout(() => {
            chatBox.scrollTo({
                top: chatBox.scrollHeight,
                behavior: 'smooth'
            });
        }, 100);
    }

    function escapeHtml(value) {
        const div = document.createElement('div');
        div.textContent = value;
        return div.innerHTML;
    }
});
