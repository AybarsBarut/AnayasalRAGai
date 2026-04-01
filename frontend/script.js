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
        
        // Show loader
        loader.classList.remove('hidden');
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
                throw new Error('Sunucu ile iletişim kurulamadı.');
            }

            const data = await response.json();
            loader.classList.add('hidden');
            appendMessage('bot', data.answer);
        } catch (error) {
            loader.classList.add('hidden');
            appendMessage('bot', `Hata: ${error.message} - Sistem şu anda çevrimdışı veya yükleniyor olabilir.`);
        }
    }

    function appendMessage(sender, text) {
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
