class DomainAnalyzerUI {
    constructor() {
        this.sessionId = localStorage.getItem('session_id') || null;
        this.isConnected = false;
        this.isAnalyzing = false;
        this.apiBaseUrl = 'http://localhost:5000/api';
        
        this.initializeElements();
        this.attachEventListeners();
        this.checkApiStatus();
    }

    initializeElements() {
        this.elements = {
            statusDot: document.getElementById('statusDot'),
            statusText: document.getElementById('statusText'),
            apiKey: document.getElementById('apiKey'),
            initializeBtn: document.getElementById('initializeBtn'),
            domainInput: document.getElementById('domainInput'),
            analyzeBtn: document.getElementById('analyzeBtn'),
            syncBtn: document.getElementById('syncBtn'),
            clearChatBtn: document.getElementById('clearChatBtn'),
            welcomeScreen: document.getElementById('welcomeScreen'),
            chatContainer: document.getElementById('chatContainer'),
            chatMessages: document.getElementById('chatMessages'),
            chatInput: document.getElementById('chatInput'),
            sendBtn: document.getElementById('sendBtn'),
            loadingIndicator: document.getElementById('loadingIndicator'),
            notification: document.getElementById('notification'),
            uploadForm: document.getElementById('uploadForm'),
            uploadSessionId: document.getElementById('uploadSessionId')
        };
    }

    attachEventListeners() {
    this.elements.initializeBtn.addEventListener('click', () => this.initializeAnalyzer());
    this.elements.analyzeBtn.addEventListener('click', () => this.analyzeDomain());
    this.elements.syncBtn.addEventListener('click', () => this.syncDomain());
    this.elements.clearChatBtn.addEventListener('click', () => this.clearChat());
    this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
    
    this.elements.chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    });

    this.elements.apiKey.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            this.initializeAnalyzer();
        }
    });

    this.elements.domainInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            this.analyzeDomain();
        }
    });

    // ✅ Add this block for upload HTML
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = new FormData(uploadForm);
            const sessionId = this.sessionId || localStorage.getItem('session_id');

            if (!sessionId) {
                alert("❌ Please initialize the analyzer first.");
                return;
            }

            formData.set('session_id', sessionId);
            const uploadSessionInput = document.getElementById('uploadSessionId');
            if (uploadSessionInput) {
                uploadSessionInput.value = sessionId;
            }


            try {
                const res = await fetch(`${this.apiBaseUrl}/upload-html`, {
                    method: 'POST',
                    body: formData
                });

                const result = await res.json();

                if (result.success) {
                    this.addMessage('assistant', 'HTML Summary', result.summary);
                    this.showNotification('HTML file analyzed successfully!', 'success');
                    this.showWelcomeScreen(false);
                    this.showChatContainer(true);
                    this.enableChatInputs();
                } else {
                    this.showNotification(result.message, 'error');

                }
            } catch (error) {
                alert("❌ Failed to upload HTML file.");
            }
        });
    }
    // 👇 ADD THIS INSIDE attachEventListeners()

const modeSelect = document.getElementById('modeSelect');
const domainInputGroup = document.getElementById('domainInputGroup');
const urlsInputGroup = document.getElementById('urlsInputGroup');
const uploadFormElement = document.getElementById('uploadForm');
const domainInput = document.getElementById('domainInput');
const urlInput = document.getElementById('urlInput');
const analyzeBtn = document.getElementById('analyzeBtn');

modeSelect.addEventListener('change', () => {
    const selectedMode = modeSelect.value;

    // Hide all
    domainInputGroup.style.display = 'none';
    urlsInputGroup.style.display = 'none';
    uploadFormElement.style.display = 'none';

    // Disable all
    domainInput.disabled = true;
    urlInput.disabled = true;
    analyzeBtn.disabled = true;

    // Show/Enable based on selection
    if (selectedMode === 'domain') {
        domainInputGroup.style.display = 'block';
        domainInput.disabled = false;
        analyzeBtn.disabled = false;
    } else if (selectedMode === 'urls') {
        urlsInputGroup.style.display = 'block';
        urlInput.disabled = false;
        analyzeBtn.disabled = false;
    } else if (selectedMode === 'upload') {
        uploadFormElement.style.display = 'block';
    }
});

}


    async checkApiStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/status`);
            if (response.ok) {
                this.showNotification('API server is running', 'success');
            }
        } catch (error) {
            this.showNotification('Cannot connect to API server. Please ensure the Flask app is running.', 'error');
        }
    }

    async initializeAnalyzer() {
        const apiKey = this.elements.apiKey.value.trim();
        
        if (!apiKey) {
            this.showNotification('Please enter your Groq API key', 'error');
            return;
        }

        this.setLoading(this.elements.initializeBtn, true);

        try {
            const response = await fetch(`${this.apiBaseUrl}/initialize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ api_key: apiKey })
            });

            const data = await response.json();

            if (data.success) {
                this.sessionId = data.session_id;
                localStorage.setItem('session_id', this.sessionId); 
                this.isConnected = true;
                this.updateConnectionStatus(true);
                this.enableDomainInputs();
                this.showNotification(data.message, 'success');
            } else {
                this.showNotification(data.message, 'error');
            }
        } catch (error) {
            this.showNotification('Error connecting to server', 'error');
        } finally {
            this.setLoading(this.elements.initializeBtn, false);
        }
    }

    async analyzeDomain() {
        const domain = this.elements.domainInput.value.trim();
        
        if (!domain) {
            this.showNotification('Please enter a domain to analyze', 'error');
            return;
        }

        if (!this.isConnected) {
            this.showNotification('Please initialize the analyzer first', 'error');
            return;
        }

        this.isAnalyzing = true;
        this.setLoading(this.elements.analyzeBtn, true);
        this.showWelcomeScreen(false);
        this.showChatContainer(true);

        // Add system message
        this.addMessage('system', 'System', `🔍 Starting analysis of ${domain}...`);

        try {
            const response = await fetch(`${this.apiBaseUrl}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    session_id: this.sessionId,
                    domain: domain 
                })
            });

            const data = await response.json();

            if (data.success) {
                this.addMessage('system', 'Analysis Complete', '✅ Domain analysis completed successfully!');
                this.addMessage('assistant', 'AI Analyst', data.summary);
                
                if (data.crawl_info) {
                    this.addMessage('assistant', 'Crawl Report', data.crawl_info);
                }

                this.enableChatInputs();
                this.elements.syncBtn.disabled = false;
                this.showNotification('Analysis completed!', 'success');
            } else {
                this.addMessage('system', 'Error', `❌ ${data.message}`);
                this.showNotification(data.message, 'error');
            }
        } catch (error) {
            this.addMessage('system', 'Error', '❌ Network error occurred');
            this.showNotification('Network error occurred', 'error');
        } finally {
            this.isAnalyzing = false;
            this.setLoading(this.elements.analyzeBtn, false);
        }
    }

    async sendMessage() {
        const message = this.elements.chatInput.value.trim();
        
        if (!message || !this.isConnected) return;

        // Add user message
        this.addMessage('user', 'You', message);
        this.elements.chatInput.value = '';
        
        // Show loading
        this.elements.loadingIndicator.classList.add('active');

        try {
            const response = await fetch(`${this.apiBaseUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    session_id: this.sessionId,
                    message: message 
                })
            });

            const data = await response.json();

            if (data.success) {
                this.addMessage('assistant', 'AI Assistant', data.response);
            } else {
                this.addMessage('system', 'Error', `❌ ${data.message}`);
            }
        } catch (error) {
            this.addMessage('system', 'Error', '❌ Network error occurred');
        } finally {
            this.elements.loadingIndicator.classList.remove('active');
        }
    }

    async syncDomain() {
        if (!this.isConnected) return;

        this.setLoading(this.elements.syncBtn, true);
        this.addMessage('system', 'System', '🔄 Syncing domain data...');

        try {
            const response = await fetch(`${this.apiBaseUrl}/sync`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ session_id: this.sessionId })
            });

            const data = await response.json();

            if (data.success) {
                this.addMessage('assistant', 'Sync Complete', data.result);
                this.showNotification('Domain synced successfully!', 'success');
            } else {
                this.addMessage('system', 'Error', `❌ ${data.message}`);
                this.showNotification(data.message, 'error');
            }
        } catch (error) {
            this.addMessage('system', 'Error', '❌ Sync failed');
            this.showNotification('Sync failed', 'error');
        } finally {
            this.setLoading(this.elements.syncBtn, false);
        }
    }

    async clearChat() {
        if (!this.isConnected) return;

        try {
            const response = await fetch(`${this.apiBaseUrl}/clear-chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ session_id: this.sessionId })
            });

            const data = await response.json();

            if (data.success) {
                this.elements.chatMessages.innerHTML = '';
                this.showNotification('Chat history cleared!', 'success');
            }
        } catch (error) {
            this.showNotification('Failed to clear chat', 'error');
        }
    }

    addMessage(type, sender, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;

        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        
        switch(type) {
            case 'user':
                avatarDiv.textContent = 'U';
                break;
            case 'assistant':
                avatarDiv.textContent = 'AI';
                break;
            case 'system':
                avatarDiv.textContent = '⚙️';
                break;
        }


        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        const senderDiv = document.createElement('h4');
        senderDiv.textContent = sender;

        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        
        // Convert markdown to HTML
        if (typeof marked !== 'undefined') {
            textDiv.innerHTML = marked.parse(content);
            if (typeof mermaid !== 'undefined') {
                mermaid.init(undefined, textDiv.querySelectorAll('code.language-mermaid'));
            }

        } else {
            textDiv.textContent = content;
        }

        contentDiv.appendChild(senderDiv);
        contentDiv.appendChild(textDiv);

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);

        this.elements.chatMessages.appendChild(messageDiv);
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
    }

    updateConnectionStatus(connected) {
        if (connected) {
            this.elements.statusDot.classList.add('connected');
            this.elements.statusText.textContent = 'Connected';
        } else {
            this.elements.statusDot.classList.remove('connected');
            this.elements.statusText.textContent = 'Not Connected';
        }
    }

    enableDomainInputs() {
        this.elements.domainInput.disabled = false;
        this.elements.analyzeBtn.disabled = false;
    }

    enableChatInputs() {
        this.elements.chatInput.disabled = false;
        this.elements.sendBtn.disabled = false;
        this.elements.clearChatBtn.disabled = false;
    }

    showWelcomeScreen(show) {
        this.elements.welcomeScreen.style.display = show ? 'flex' : 'none';
    }

    showChatContainer(show) {
        this.elements.chatContainer.style.display = show ? 'flex' : 'none';
    }

    setLoading(button, loading) {
        if (loading) {
            button.disabled = true;
            button.innerHTML = '<div class="spinner"></div> Loading...';
        } else {
            button.disabled = false;
            // Restore original button text
            if (button === this.elements.initializeBtn) {
                button.innerHTML = '🚀 Initialize Analyzer';
            } else if (button === this.elements.analyzeBtn) {
                button.innerHTML = '🔍 Analyze Website';
            } else if (button === this.elements.syncBtn) {
                button.innerHTML = '🔄 Sync Updates';
            }
        }
    }

    showNotification(message, type = 'info') {
        this.elements.notification.textContent = message;
        this.elements.notification.className = `notification ${type}`;
        this.elements.notification.classList.add('show');

        setTimeout(() => {
            this.elements.notification.classList.remove('show');
        }, 4000);
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new DomainAnalyzerUI();
});