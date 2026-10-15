"""ChatGPT / Claude-style conversational interface for RailDrishti.

Features:
- Native Speech-to-Text with microphone button on the right side of the input box
- Live real-time speech transcription writing directly into the text input
- Full conversational AI powered by Sarvam 105B Indic LLM via FastAPI backend
- Voice playback on every response with Sarvam Bulbul TTS & Web Speech API fallback
- One-click starter suggestions, markdown rendering, copy buttons, and clean IR aesthetics
"""
from __future__ import annotations


def get_chatgpt_html(api_port: int = 8001) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RailDrishti AI Chat</title>
<style>
  :root {{
    --ir-navy: #0B2A66;
    --ir-navy-hover: #143A85;
    --ir-red: #C8102E;
    --ir-red-light: #FFEBEE;
    --bg: #F7F5F0;
    --surface: #FFFFFF;
    --surface-muted: #EFEBE2;
    --border: #E1DACB;
    --border-focus: #0B2A66;
    --ink: #1A2430;
    --muted: #5D6675;
    --success: #1E7A46;
  }}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: var(--bg);
    color: var(--ink);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}

  /* Top Control Bar */
  .chat-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 18px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }}

  .chat-header-title {{
    display: flex;
    align-items: center;
    gap: 10px;
  }}

  .chat-logo {{
    width: 32px;
    height: 32px;
    background: var(--ir-navy);
    color: #fff;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
  }}

  .chat-title-text {{
    font-size: 1rem;
    font-weight: 700;
    color: var(--ir-navy);
  }}

  .chat-status-badge {{
    font-size: 0.75rem;
    background: #E8F5E9;
    color: var(--success);
    padding: 3px 8px;
    border-radius: 12px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }}

  .chat-status-badge .dot {{
    width: 6px;
    height: 6px;
    background: var(--success);
    border-radius: 50%;
  }}

  .chat-header-actions {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}

  .chat-select {{
    padding: 6px 10px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--ink);
    cursor: pointer;
    outline: none;
  }}

  .chat-select:focus {{
    border-color: var(--ir-navy);
  }}

  .btn-icon {{
    background: transparent;
    border: 1px solid var(--border);
    padding: 6px 10px;
    border-radius: 8px;
    color: var(--muted);
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s;
  }}

  .btn-icon:hover {{
    background: var(--surface-muted);
    color: var(--ink);
  }}

  /* Messages Area */
  .chat-messages {{
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 18px;
    scroll-behavior: smooth;
  }}

  /* Welcome Screen */
  .welcome-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    max-width: 650px;
    margin: 20px auto;
    box-shadow: 0 4px 16px rgba(0,0,0,0.03);
  }}

  .welcome-card h2 {{
    font-size: 1.35rem;
    color: var(--ir-navy);
    margin-bottom: 8px;
    font-weight: 700;
  }}

  .welcome-card p {{
    font-size: 0.92rem;
    color: var(--muted);
    line-height: 1.5;
    margin-bottom: 20px;
  }}

  .starter-chips {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
  }}

  .chip {{
    background: var(--surface-muted);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 7px 14px;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--ink);
    cursor: pointer;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}

  .chip:hover {{
    background: var(--ir-navy);
    color: #fff;
    border-color: var(--ir-navy);
    transform: translateY(-1px);
  }}

  /* Message Bubbles */
  .msg-row {{
    display: flex;
    gap: 12px;
    max-width: 820px;
    width: 100%;
    margin: 0 auto;
  }}

  .msg-row.user {{
    justify-content: flex-end;
  }}

  .msg-avatar {{
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    font-weight: 700;
    flex-shrink: 0;
  }}

  .msg-avatar.bot {{
    background: var(--ir-navy);
    color: #fff;
  }}

  .msg-avatar.user {{
    background: #4A5568;
    color: #fff;
  }}

  .msg-content {{
    max-width: calc(100% - 50px);
  }}

  .msg-bubble {{
    padding: 12px 18px;
    border-radius: 16px;
    font-size: 0.93rem;
    line-height: 1.6;
    word-break: break-word;
  }}

  .msg-row.user .msg-bubble {{
    background: var(--ir-navy);
    color: #FFFFFF;
    border-bottom-right-radius: 4px;
  }}

  .msg-row.bot .msg-bubble {{
    background: var(--surface);
    color: var(--ink);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
  }}

  .msg-bubble p {{
    margin-bottom: 8px;
  }}

  .msg-bubble p:last-child {{
    margin-bottom: 0;
  }}

  .msg-bubble ul, .msg-bubble ol {{
    margin: 8px 0 8px 20px;
  }}

  .msg-bubble li {{
    margin-bottom: 4px;
  }}

  .msg-bubble strong {{
    color: inherit;
    font-weight: 600;
  }}

  .msg-bubble table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 0.85rem;
  }}

  .msg-bubble th, .msg-bubble td {{
    border: 1px solid var(--border);
    padding: 6px 10px;
    text-align: left;
  }}

  .msg-bubble th {{
    background: var(--surface-muted);
    font-weight: 600;
  }}

  .msg-actions {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 6px;
    padding-left: 4px;
  }}

  .action-btn {{
    background: transparent;
    border: none;
    color: var(--muted);
    font-size: 0.78rem;
    font-weight: 500;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 6px;
    border-radius: 6px;
    transition: all 0.2s;
  }}

  .action-btn:hover {{
    color: var(--ir-navy);
    background: var(--surface-muted);
  }}

  .action-btn.playing {{
    color: var(--ir-red);
    font-weight: 600;
  }}

  /* Thinking Indicator */
  .thinking-bubble {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 12px 18px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    width: fit-content;
  }}

  .thinking-dot {{
    width: 7px;
    height: 7px;
    background: var(--ir-navy);
    border-radius: 50%;
    animation: thinking 1.4s infinite ease-in-out both;
  }}

  .thinking-dot:nth-child(1) {{ animation-delay: -0.32s; }}
  .thinking-dot:nth-child(2) {{ animation-delay: -0.16s; }}

  @keyframes thinking {{
    0%, 80%, 100% {{ transform: scale(0); opacity: 0.4; }}
    40% {{ transform: scale(1); opacity: 1; }}
  }}

  /* Bottom Input Bar - ChatGPT / Claude style */
  .chat-input-container {{
    padding: 12px 20px 18px;
    background: linear-gradient(to top, var(--bg) 80%, rgba(247, 245, 240, 0));
    flex-shrink: 0;
  }}

  .listening-banner {{
    max-width: 820px;
    margin: 0 auto 8px;
    background: var(--ir-red-light);
    border: 1px solid var(--ir-red);
    color: var(--ir-red);
    padding: 6px 14px;
    border-radius: 10px;
    font-size: 0.82rem;
    font-weight: 600;
    display: none;
    align-items: center;
    justify-content: space-between;
    animation: pulse-banner 2s infinite;
  }}

  @keyframes pulse-banner {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.75; }}
  }}

  .chat-box-wrapper {{
    max-width: 820px;
    margin: 0 auto;
    position: relative;
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 28px;
    box-shadow: 0 4px 20px rgba(11, 42, 102, 0.06);
    display: flex;
    align-items: center;
    padding: 4px 8px 4px 18px;
    transition: all 0.2s ease;
  }}

  .chat-box-wrapper:focus-within {{
    border-color: var(--border-focus);
    box-shadow: 0 4px 24px rgba(11, 42, 102, 0.12);
  }}

  .chat-textarea {{
    flex: 1;
    border: none;
    outline: none;
    background: transparent;
    font-size: 0.95rem;
    color: var(--ink);
    padding: 10px 4px;
    resize: none;
    font-family: inherit;
    line-height: 1.45;
    max-height: 140px;
    min-height: 24px;
  }}

  .chat-textarea::placeholder {{
    color: #9CA3AF;
  }}

  .chat-box-actions {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  /* Microphone Button */
  .btn-mic {{
    width: 38px;
    height: 38px;
    border-radius: 50%;
    border: none;
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
  }}

  .btn-mic:hover {{
    background: var(--surface-muted);
    color: var(--ir-navy);
  }}

  .btn-mic.recording {{
    background: var(--ir-red);
    color: #fff;
    animation: mic-pulse 1.4s infinite;
  }}

  @keyframes mic-pulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(200, 16, 46, 0.6); }}
    70% {{ box-shadow: 0 0 0 10px rgba(200, 16, 46, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(200, 16, 46, 0); }}
  }}

  /* Send Button */
  .btn-send {{
    width: 38px;
    height: 38px;
    border-radius: 50%;
    border: none;
    background: var(--ir-navy);
    color: #FFFFFF;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
  }}

  .btn-send:hover:not(:disabled) {{
    background: var(--ir-navy-hover);
    transform: scale(1.04);
  }}

  .btn-send:disabled {{
    background: #E5E7EB;
    color: #9CA3AF;
    cursor: not-allowed;
  }}

  .chat-hint {{
    text-align: center;
    font-size: 0.74rem;
    color: var(--muted);
    margin-top: 6px;
  }}
</style>
</head>
<body>

<!-- Header -->
<div class="chat-header">
  <div class="chat-header-title">
    <div class="chat-logo">IR</div>
    <div>
      <div class="chat-title-text">RailBot — Conversational Assistant</div>
      <div class="chat-status-badge">
        <span class="dot"></span> Sarvam 105B Indic LLM · Active
      </div>
    </div>
  </div>
  <div class="chat-header-actions">
    <select id="langSelect" class="chat-select" title="Reply Language">
      <option value="">Auto-Detect Language</option>
      <option value="en">English (Indian)</option>
      <option value="hi">हिन्दी · Hindi</option>
      <option value="ta">தமிழ் · Tamil</option>
      <option value="te">తెలుగు · Telugu</option>
      <option value="bn">বাংলা · Bengali</option>
      <option value="mr">मराठी · Marathi</option>
    </select>
    <button id="clearChatBtn" class="btn-icon" title="Clear Conversation">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
      </svg>
      Clear
    </button>
  </div>
</div>

<!-- Chat Messages Stream -->
<div id="chatMessages" class="chat-messages">
  <div id="welcomeScreen" class="welcome-card">
    <h2>🚆 How can I assist your journey today?</h2>
    <p>Ask anything about Tatkal rules, waitlist confirmation chances, train delays, cancellation refunds, or plan a multi-city journey — by typing or speaking directly.</p>
    <div class="starter-chips">
      <button class="chip" onclick="applyPrompt('What are the rules and timings for Tatkal booking for AC and Non-AC?')">⚡ Tatkal booking rules & timing</button>
      <button class="chip" onclick="applyPrompt('Will WL 15 on an Express train confirm for travel in 15 days?')">🎫 Will WL 15 confirm?</button>
      <button class="chip" onclick="applyPrompt('What is the refund rule if a train is delayed by more than 3 hours?')">💰 Refund policy for train delay</button>
      <button class="chip" onclick="applyPrompt('Can I carry my pet dog in a 3AC coach?')">🐕 Pet travel in 3AC</button>
      <button class="chip" onclick="applyPrompt('Plan a journey from Pune to Delhi with cheapest AC class')">🧭 Plan Pune to Delhi trip</button>
    </div>
  </div>
</div>

<!-- Bottom Input Bar -->
<div class="chat-input-container">
  <div id="listeningBanner" class="listening-banner">
    <span>🔴 Listening... Speak into your microphone</span>
    <button onclick="stopVoiceRecognition()" style="background:none;border:none;color:inherit;cursor:pointer;font-weight:700;font-size:0.8rem">Stop ✕</button>
  </div>

  <div class="chat-box-wrapper">
    <textarea id="chatInput" class="chat-textarea" rows="1" placeholder="Message RailDrishti or tap the mic to speak..."></textarea>
    <div class="chat-box-actions">
      <!-- Microphone Button on the Right Side -->
      <button id="micBtn" class="btn-mic" title="Click to speak (voice dictation)" onclick="toggleVoiceRecognition()">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"></path>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
          <line x1="12" y1="19" x2="12" y2="22"></line>
        </svg>
      </button>

      <!-- Send Button -->
      <button id="sendBtn" class="btn-send" title="Send message" onclick="sendMessage()" disabled>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <line x1="12" y1="19" x2="12" y2="5"></line>
          <polyline points="5 12 12 5 19 12"></polyline>
        </svg>
      </button>
    </div>
  </div>
  <div class="chat-hint">RailDrishti AI synthesizes real-time Indian Railways rules & machine learning models.</div>
</div>

<script>
  let API_BASE = "http://127.0.0.1:{api_port}";
  try {
    if (window.parent && window.parent.location && window.parent.location.hostname) {
      API_BASE = window.parent.location.protocol + "//" + window.parent.location.hostname + ":{api_port}";
    }
  } catch (e) {
    API_BASE = "http://127.0.0.1:{api_port}";
  }
  const chatMessages = document.getElementById("chatMessages");
  const welcomeScreen = document.getElementById("welcomeScreen");
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const micBtn = document.getElementById("micBtn");
  const listeningBanner = document.getElementById("listeningBanner");
  const langSelect = document.getElementById("langSelect");
  const clearChatBtn = document.getElementById("clearChatBtn");

  let conversationHistory = [];
  let isRecording = false;
  let recognition = null;
  let activeAudio = null;

  // Auto-resize textarea
  chatInput.addEventListener("input", () => {{
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
    sendBtn.disabled = !chatInput.value.trim();
  }});

  // Enter to send (Shift+Enter for newline)
  chatInput.addEventListener("keydown", (e) => {{
    if (e.key === "Enter" && !e.shiftKey) {{
      e.preventDefault();
      if (!sendBtn.disabled) {{
        sendMessage();
      }}
    }}
  }});

  function applyPrompt(text) {{
    chatInput.value = text;
    chatInput.dispatchEvent(new Event("input"));
    sendMessage();
  }}

  // -------------------------------------------------------------------------
  // Speech-to-Text (Microphone Voice Typing)
  // -------------------------------------------------------------------------
  function initSpeechRecognition() {{
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {{
      alert("Voice speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.");
      return null;
    }}
    const sr = new SpeechRecognition();
    sr.continuous = true;
    sr.interimResults = true;

    sr.onstart = () => {{
      isRecording = true;
      micBtn.classList.add("recording");
      listeningBanner.style.display = "flex";
    }};

    sr.onresult = (event) => {{
      let fullText = "";
      for (let i = 0; i < event.results.length; i++) {{
        fullText += event.results[i][0].transcript + " ";
      }}
      chatInput.value = fullText.trim();
      chatInput.dispatchEvent(new Event("input"));
    }};

    sr.onerror = (event) => {{
      console.warn("Speech recognition error:", event.error);
      stopVoiceRecognition();
    }};

    sr.onend = () => {{
      stopVoiceRecognition();
    }};

    return sr;
  }}

  function toggleVoiceRecognition() {{
    if (isRecording) {{
      stopVoiceRecognition();
    }} else {{
      startVoiceRecognition();
    }}
  }}

  function startVoiceRecognition() {{
    if (!recognition) {{
      recognition = initSpeechRecognition();
    }}
    if (!recognition) return;

    const lang = langSelect.value;
    if (lang === "hi") recognition.lang = "hi-IN";
    else if (lang === "ta") recognition.lang = "ta-IN";
    else if (lang === "te") recognition.lang = "te-IN";
    else if (lang === "bn") recognition.lang = "bn-IN";
    else if (lang === "mr") recognition.lang = "mr-IN";
    else recognition.lang = "en-IN";

    try {{
      recognition.start();
    }} catch (e) {{
      console.warn(e);
    }}
  }}

  function stopVoiceRecognition() {{
    isRecording = false;
    micBtn.classList.remove("recording");
    listeningBanner.style.display = "none";
    if (recognition) {{
      try {{ recognition.stop(); }} catch (e) {{}}
    }}
  }}

  // -------------------------------------------------------------------------
  // Sending & Receiving Messages
  // -------------------------------------------------------------------------
  async function sendMessage() {{
    const query = chatInput.value.trim();
    if (!query) return;

    stopVoiceRecognition();
    if (welcomeScreen) welcomeScreen.style.display = "none";

    // Append User Message
    appendMessage("user", query);
    conversationHistory.push({{ role: "user", content: query }});

    chatInput.value = "";
    chatInput.style.height = "auto";
    sendBtn.disabled = true;

    // Show Thinking indicator
    const thinkingRow = showThinkingIndicator();

    try {{
      const resp = await fetch(API_BASE + "/api/chat", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{
          query: query,
          language: langSelect.value || null,
          history: conversationHistory
        }})
      }});

      thinkingRow.remove();

      if (!resp.ok) {{
        throw new Error("HTTP " + resp.status);
      }}

      const data = await resp.json();
      const answer = data.answer || "No response received.";
      conversationHistory.push({{ role: "assistant", content: answer }});

      appendMessage("bot", answer, data);

    }} catch (err) {{
      thinkingRow.remove();
      appendMessage("bot", "⚠️ Could not connect to the RailDrishti server (" + err.message + "). Please make sure the service is running on port {api_port}.");
    }}
  }}

  function showThinkingIndicator() {{
    const row = document.createElement("div");
    row.className = "msg-row bot";
    row.innerHTML = `
      <div class="msg-avatar bot">IR</div>
      <div class="thinking-bubble">
        <div class="thinking-dot"></div>
        <div class="thinking-dot"></div>
        <div class="thinking-dot"></div>
      </div>
    `;
    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return row;
  }}

  function appendMessage(role, text, meta) {{
    const row = document.createElement("div");
    row.className = "msg-row " + role;

    const avatarHtml = role === "user"
      ? '<div class="msg-avatar user">👤</div>'
      : '<div class="msg-avatar bot">IR</div>';

    const formattedContent = role === "bot" ? renderMarkdown(text) : escapeHtml(text);

    let actionsHtml = "";
    if (role === "bot") {{
      actionsHtml = `
        <div class="msg-actions">
          <button class="action-btn" onclick="playVoice(this, decodeURIComponent('${{encodeURIComponent(text)}}'))">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
              <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
            </svg>
            Listen
          </button>
          <button class="action-btn" onclick="copyText(this, decodeURIComponent('${{encodeURIComponent(text)}}'))">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            Copy
          </button>
          ${{meta && meta.model ? `<span style="font-size:0.72rem;color:var(--muted);margin-left:6px">⚡ ${{meta.model}} (${{meta.latency_ms || 0}}ms)</span>` : ''}}
        </div>
      `;
    }}

    row.innerHTML = `
      ${{role === "bot" ? avatarHtml : ""}}
      <div class="msg-content">
        <div class="msg-bubble">${{formattedContent}}</div>
        ${{actionsHtml}}
      </div>
      ${{role === "user" ? avatarHtml : ""}}
    `;

    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }}

  // -------------------------------------------------------------------------
  // Voice Output (Sarvam TTS / Web Speech)
  // -------------------------------------------------------------------------
  async function playVoice(btn, text) {{
    if (activeAudio) {{
      activeAudio.pause();
      activeAudio = null;
      document.querySelectorAll(".action-btn.playing").forEach(b => {{
        b.classList.remove("playing");
        b.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
          </svg>
          Listen
        `;
      }});
      return;
    }}

    btn.classList.add("playing");
    btn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
        <rect x="6" y="4" width="4" height="16"></rect>
        <rect x="14" y="4" width="4" height="16"></rect>
      </svg>
      Stop
    `;

    try {{
      // Try Sarvam TTS endpoint
      const resp = await fetch(API_BASE + "/api/tts", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ text: text, language: langSelect.value || "en" }})
      }});

      if (resp.ok) {{
        const data = await resp.json();
        if (data.audio_url) {{
          activeAudio = new Audio(data.audio_url);
          activeAudio.onended = () => {{
            activeAudio = null;
            btn.classList.remove("playing");
            btn.innerHTML = `
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
              </svg>
              Listen
            `;
          }};
          activeAudio.play();
          return;
        }}
      }}
    }} catch (e) {{
      console.warn("Sarvam TTS failed, falling back to Web Speech:", e);
    }}

    // Browser SpeechSynthesis fallback
    if ("speechSynthesis" in window) {{
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text.replace(/[*_#`]/g, ""));
      u.lang = langSelect.value === "hi" ? "hi-IN" : "en-IN";
      u.onend = () => {{
        btn.classList.remove("playing");
        btn.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
          </svg>
          Listen
        `;
      }};
      window.speechSynthesis.speak(u);
    }}
  }}

  function copyText(btn, text) {{
    navigator.clipboard.writeText(text).then(() => {{
      const orig = btn.innerHTML;
      btn.innerHTML = "✓ Copied";
      setTimeout(() => {{ btn.innerHTML = orig; }}, 1600);
    }});
  }}

  clearChatBtn.addEventListener("click", () => {{
    conversationHistory = [];
    chatMessages.innerHTML = "";
    if (welcomeScreen) {{
      chatMessages.appendChild(welcomeScreen);
      welcomeScreen.style.display = "block";
    }}
    if (activeAudio) {{
      activeAudio.pause();
      activeAudio = null;
    }}
  }});

  // Simple Markdown Parser
  function escapeHtml(text) {{
    const div = document.createElement("div");
    div.innerText = text;
    return div.innerHTML;
  }}

  function renderMarkdown(md) {{
    if (!md) return "";
    let html = escapeHtml(md);

    // Code blocks
    html = html.replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold
    html = html.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
    // Italic
    html = html.replace(/\\*(.*?)\\*/g, '<em>$1</em>');
    // Headings
    html = html.replace(/^### (.*$)/gim, '<h4 style="color:var(--ir-navy);margin:10px 0 4px">$1</h4>');
    html = html.replace(/^## (.*$)/gim, '<h3 style="color:var(--ir-navy);margin:12px 0 6px">$1</h3>');
    html = html.replace(/^# (.*$)/gim, '<h2 style="color:var(--ir-navy);margin:14px 0 8px">$1</h2>');
    // Bullet lists
    html = html.replace(/^\\* (.*$)/gim, '<li>$1</li>');
    html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\\/li>)/s, '<ul>$1</ul>');
    // Line breaks
    html = html.replace(/\\n\\n/g, '<p></p>');
    html = html.replace(/\\n/g, '<br>');

    return html;
  }}
</script>
</body>
</html>
"""
