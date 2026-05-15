document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const loader = document.getElementById('loader');

    // Make enter key submit
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // Add user message to UI
        appendMessage('user', text);
        userInput.value = '';
        
        setLoading(true);
        chatBox.scrollTop = chatBox.scrollHeight;

        try {
            // In a real deployed app, the host would match the frontend host
            // Since we mount this folder in FastAPI, we can use relative path '/chat'
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: text })
            });

            if (!response.ok) {
                throw new Error(await getErrorMessage(response));
            }

            const data = await response.json();
            appendMessage('bot', data.answer, {
                confidence: data.confidence,
                citations: data.citations || [],
                reviewNotes: data.review_notes || []
            });
        } catch (error) {
            appendMessage('bot', `Hata: ${error.message} - Sistem şu anda çevrimdışı veya yükleniyor olabilir.`);
        } finally {
            setLoading(false);
        }
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
    }

    function confidenceLabel(confidence) {
        const labels = {
            verified: 'Alıntı doğrulandı',
            source_grounded: 'Kaynaklı yanıt',
            needs_review: 'Kontrol gerekli'
        };
        return labels[confidence] || labels.needs_review;
    }

    function appendGuardrailMetadata(container, meta = {}) {
        const hasCitations = Array.isArray(meta.citations) && meta.citations.length > 0;
        const hasNotes = Array.isArray(meta.reviewNotes) && meta.reviewNotes.length > 0;
        if (!meta.confidence && !hasCitations && !hasNotes) return;

        const panel = document.createElement('div');
        panel.className = 'guardrail-panel';

        if (meta.confidence) {
            const badge = document.createElement('div');
            badge.className = `confidence-badge confidence-${meta.confidence}`;
            badge.textContent = confidenceLabel(meta.confidence);
            panel.appendChild(badge);
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

            meta.reviewNotes.slice(0, 3).forEach((note) => {
                const item = document.createElement('li');
                item.textContent = note;
                notes.appendChild(item);
            });

            panel.appendChild(notes);
        }

        container.appendChild(panel);
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
            // Parse Markdown for bot responses (requires marked.js included in HTML)
            content.innerHTML = marked.parse(text);
            appendGuardrailMetadata(content, meta);
        } else {
            // Raw text for user
            content.innerText = text;
        }

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);

        chatBox.appendChild(msgDiv);
        
        // Auto scroll to bottom smoothly
        setTimeout(() => {
            chatBox.scrollTo({
                top: chatBox.scrollHeight,
                behavior: 'smooth'
            });
        }, 100);
    }
});
