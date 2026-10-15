"""Standalone conversational AI web application for RailDrishti.

Inspired by ChatGPT Voice and AI Interviewer platforms:
- Continuous real-time voice recognition (speech-to-text directly writing into the input)
- Autonomous voice output (text-to-speech auto-play with Sarvam Bulbul V3)
- Live audio wave visualizer
- Deep Indian Railways reasoning via Sarvam 105B Indic LLM + RailDrishti ML models
"""
from __future__ import annotations


def get_chat_page_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>RailDrishti · Conversational AI Companion</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --ir-navy: #0B2A66;
    --ir-navy-dark: #071D49;
    --ir-navy-light: #143A85;
    --ir-red: #C8102E;
    --ir-red-hover: #A00C24;
    --ir-red-light: rgba(200, 16, 46, 0.12);
    --bg-base: #F7F5F0;
    --bg-mesh: radial-gradient(at 0% 0%, rgba(11, 42, 102, 0.08) 0px, transparent 50%),
               radial-gradient(at 100% 0%, rgba(200, 16, 46, 0.06) 0px, transparent 50%),
               radial-gradient(at 50% 100%, rgba(11, 42, 102, 0.05) 0px, transparent 50%), #F7F5F0;
    --panel: #FFFFFF;
    --panel-hover: #FDFDFD;
    --panel-muted: #EFEBE2;
    --panel-input: #FFFFFF;
    --line: #E1DACB;
    --line-focus: #0B2A66;
    --text: #1A2430;
    --text-heading: #0B2A66;
    --muted: #5D6675;
    --muted-dark: #8C96A5;
    --good: #1E7A46;
    --good-bg: rgba(30, 122, 70, 0.1);
    --shadow-sm: 0 2px 8px rgba(11, 42, 102, 0.05);
    --shadow-lg: 0 12px 36px rgba(11, 42, 102, 0.09);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; min-height: 100vh; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg-mesh);
    color: var(--text);
    display: flex;
    flex-direction: column;
    align-items: center;
    overflow: hidden;
  }

  /* Header */
  header {
    width: 100%;
    max-width: 980px;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
    text-decoration: none;
    color: inherit;
  }

  .logo-box {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: var(--ir-navy);
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 15px;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 12px rgba(11, 42, 102, 0.2);
  }

  .brand-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 17px;
    font-weight: 800;
    color: var(--ir-navy);
    letter-spacing: -0.3px;
    line-height: 1.2;
  }

  .brand-sub {
    font-size: 11px;
    font-weight: 600;
    color: var(--muted);
    letter-spacing: 0.3px;
    text-transform: uppercase;
  }

  .header-controls {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--good-bg);
    border: 1px solid rgba(30, 122, 70, 0.25);
    color: var(--good);
    padding: 5px 11px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
  }

  .pill-dot {
    width: 7px;
    height: 7px;
    background: var(--good);
    border-radius: 50%;
    animation: livePulse 1.8s infinite;
  }

  @keyframes livePulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
  }

  .select-lang {
    padding: 6px 12px;
    border-radius: 10px;
    background: var(--panel);
    border: 1px solid var(--line);
    color: var(--text);
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    outline: none;
    transition: all 0.2s ease;
  }

  .select-lang:focus {
    border-color: var(--ir-navy);
  }

  .btn-header {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 10px;
    background: var(--panel);
    border: 1px solid var(--line);
    color: var(--text);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    text-decoration: none;
    transition: all 0.2s ease;
  }

  .btn-header:hover {
    background: var(--panel-muted);
    border-color: var(--ir-navy);
  }

  /* Main Chat Container */
  main {
    width: 100%;
    max-width: 980px;
    padding: 0 24px 20px;
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }

  .chat-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 20px;
    box-shadow: var(--shadow-lg);
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }

  /* Chat Messages Feed */
  #log {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 18px;
    scroll-behavior: smooth;
  }

  /* Welcome Hero Card */
  .welcome-box {
    text-align: center;
    padding: 24px 16px;
    max-width: 680px;
    margin: auto;
  }

  .welcome-icon {
    font-size: 38px;
    margin-bottom: 12px;
  }

  .welcome-box h1 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: var(--ir-navy);
    letter-spacing: -0.4px;
    margin-bottom: 8px;
  }

  .welcome-box p {
    font-size: 13.5px;
    color: var(--muted);
    line-height: 1.55;
    margin-bottom: 20px;
  }

  .chips-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
  }

  .chip-btn {
    background: var(--panel-muted);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 7px 14px;
    font-size: 12.5px;
    font-weight: 500;
    color: var(--text);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .chip-btn:hover {
    background: var(--ir-navy);
    color: #FFFFFF;
    border-color: var(--ir-navy);
    transform: translateY(-1px);
  }

  /* Messages */
  .msg {
    display: flex;
    gap: 12px;
    max-width: 840px;
    width: 100%;
    margin: 0 auto;
    animation: fadeIn 0.25s ease-out;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .msg.me {
    justify-content: flex-end;
  }

  .avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 12px;
    flex-shrink: 0;
  }

  .msg.bot .avatar {
    background: var(--ir-navy);
    color: #FFFFFF;
    box-shadow: 0 2px 8px rgba(11, 42, 102, 0.25);
  }

  .msg.me .avatar {
    background: #475569;
    color: #FFFFFF;
  }

  .bubble-wrap {
    max-width: calc(100% - 52px);
    display: flex;
    flex-direction: column;
  }

  .bubble-author {
    font-size: 11px;
    font-weight: 700;
    color: var(--muted);
    margin-bottom: 4px;
    letter-spacing: 0.3px;
  }

  .msg.me .bubble-author {
    text-align: right;
  }

  .bubble {
    padding: 13px 18px;
    border-radius: 16px;
    font-size: 14px;
    line-height: 1.6;
    word-break: break-word;
  }

  .msg.me .bubble {
    background: var(--ir-navy);
    color: #FFFFFF;
    border-bottom-right-radius: 4px;
    box-shadow: 0 2px 10px rgba(11, 42, 102, 0.15);
  }

  .msg.bot .bubble {
    background: #FFFFFF;
    color: var(--text);
    border: 1px solid var(--line);
    border-bottom-left-radius: 4px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
  }

  .bubble p { margin-bottom: 8px; }
  .bubble p:last-child { margin-bottom: 0; }
  .bubble ul, .bubble ol { margin: 8px 0 8px 20px; }
  .bubble li { margin-bottom: 4px; }
  .bubble strong { font-weight: 700; color: inherit; }
  .bubble table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 12.5px;
  }
  .bubble th, .bubble td {
    border: 1px solid var(--line);
    padding: 6px 10px;
    text-align: left;
  }
  .bubble th {
    background: var(--panel-muted);
    font-weight: 700;
  }

  /* Actions below bot bubbles */
  .bubble-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 6px;
    padding-left: 2px;
  }

  .action-btn {
    background: transparent;
    border: 1px solid transparent;
    color: var(--muted);
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    border-radius: 6px;
    transition: all 0.2s ease;
  }

  .action-btn:hover {
    color: var(--ir-navy);
    background: var(--panel-muted);
    border-color: var(--line);
  }

  .action-btn.active-speak {
    color: var(--ir-red);
    background: var(--ir-red-light);
    border-color: rgba(200, 16, 46, 0.3);
  }

  /* Audio Wave Visualizer */
  .audio-wave-container {
    display: none;
    align-items: center;
    justify-content: center;
    gap: 4px;
    padding: 10px 0 6px;
    background: var(--panel-muted);
    border-top: 1px solid var(--line);
  }

  .wave-bar {
    width: 4px;
    height: 18px;
    background: var(--ir-red);
    border-radius: 4px;
    animation: waveBounce 0.9s infinite ease-in-out;
  }

  .wave-bar:nth-child(2) { animation-delay: 0.15s; height: 26px; }
  .wave-bar:nth-child(3) { animation-delay: 0.3s; height: 32px; }
  .wave-bar:nth-child(4) { animation-delay: 0.45s; height: 22px; }
  .wave-bar:nth-child(5) { animation-delay: 0.6s; height: 14px; }

  @keyframes waveBounce {
    0%, 100% { transform: scaleY(0.4); opacity: 0.4; }
    50% { transform: scaleY(1.2); opacity: 1; }
  }

  /* Composer Card */
  .composer-card {
    padding: 14px 20px 16px;
    background: var(--panel);
    border-top: 1px solid var(--line);
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex-shrink: 0;
  }

  .composer {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #FFFFFF;
    border: 1.5px solid var(--line);
    border-radius: 16px;
    padding: 5px 8px 5px 16px;
    transition: all 0.2s ease;
    box-shadow: 0 2px 10px rgba(11, 42, 102, 0.04);
  }

  .composer:focus-within {
    border-color: var(--ir-navy);
    box-shadow: 0 0 0 3px rgba(11, 42, 102, 0.12);
  }

  textarea#input {
    flex: 1;
    border: none;
    outline: none;
    background: transparent;
    font-size: 14.5px;
    color: var(--text);
    font-family: inherit;
    resize: none;
    line-height: 1.45;
    min-height: 24px;
    max-height: 140px;
    padding: 8px 0;
  }

  textarea#input::placeholder {
    color: #94A3B8;
  }

  /* Prominent Mic Button on the Right Side */
  .btn-mic-prominent {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    border: 1.5px solid var(--line);
    background: var(--panel-muted);
    color: var(--ir-navy);
    font-size: 18px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
    flex-shrink: 0;
  }

  .btn-mic-prominent:hover {
    background: var(--ir-navy);
    color: #FFFFFF;
    border-color: var(--ir-navy);
    transform: scale(1.05);
  }

  .btn-mic-prominent.recording {
    background: var(--ir-red);
    color: #FFFFFF;
    border-color: var(--ir-red);
    animation: micPulse 1.2s infinite;
  }

  @keyframes micPulse {
    0% { box-shadow: 0 0 0 0 rgba(200, 16, 46, 0.7); }
    70% { box-shadow: 0 0 0 12px rgba(200, 16, 46, 0); }
    100% { box-shadow: 0 0 0 0 rgba(200, 16, 46, 0); }
  }

  .btn-send-prominent {
    height: 42px;
    padding: 0 16px;
    border-radius: 12px;
    border: none;
    background: var(--ir-navy);
    color: #FFFFFF;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 13.5px;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.2s ease;
    flex-shrink: 0;
  }

  .btn-send-prominent:hover:not(:disabled) {
    background: var(--ir-navy-light);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(11, 42, 102, 0.2);
  }

  .btn-send-prominent:disabled {
    background: #CBD5E1;
    color: #94A3B8;
    cursor: not-allowed;
  }

  .composer-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 4px;
    font-size: 12px;
  }

  .status-text {
    color: var(--muted);
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .status-pulse {
    width: 8px;
    height: 8px;
    background: var(--ir-red);
    border-radius: 50%;
    animation: pulse 1s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.3; transform: scale(0.8); }
  }

  .btn-clear {
    background: transparent;
    border: none;
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-clear:hover {
    color: var(--ir-red);
  }
</style>
</head>
<body>

<!-- Header -->
<header>
  <div class="brand">
    <div class="logo-box">IR</div>
    <div>
      <div class="brand-title">RailDrishti AI</div>
      <div class="brand-sub">Conversational Travel Companion</div>
    </div>
  </div>

  <div class="header-controls">
    <div class="pill-badge">
      <span class="pill-dot"></span>
      <span>Sarvam 105B Indic LLM + Studio Voice</span>
    </div>

    <label style="display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;cursor:pointer;user-select:none;">
      <input type="checkbox" id="voiceToggle" checked style="accent-color:var(--ir-navy);">
      <span>Auto-Speak Voice</span>
    </label>

    <select id="langSelect" class="select-lang" title="Select Conversation Language">
      <option value="">Auto-Detect Language</option>
      <option value="en">English (Indian)</option>
      <option value="hi">हिन्दी · Hindi</option>
      <option value="ta">தமிழ் · Tamil</option>
      <option value="te">తెలుగు · Telugu</option>
      <option value="bn">বাংলা · Bengali</option>
      <option value="mr">मराठी · Marathi</option>
    </select>
  </div>
</header>

<!-- Main Container -->
<main>
  <div class="chat-card">
    <div id="log">
      <div id="welcomeScreen" class="welcome-box">
        <div class="welcome-icon">🚆</div>
        <h1>Welcome to RailDrishti AI</h1>
        <p>Talk directly or type your message. RailDrishti provides live conversational answers with voice playback, Tatkal rules, waitlist confirmation chances, refund policies, and journey planning.</p>
        <div class="chips-container">
          <button class="chip-btn" onclick="quickPrompt('What are the rules and timings for Tatkal booking for AC and Non-AC?')">⚡ Tatkal booking timings & rules</button>
          <button class="chip-btn" onclick="quickPrompt('Will WL 15 on an Express train confirm for travel in 15 days?')">🎫 Will WL 15 confirm?</button>
          <button class="chip-btn" onclick="quickPrompt('What is the refund rule if my train is delayed by more than 3 hours?')">💰 Refund for 3+ hr delay</button>
          <button class="chip-btn" onclick="quickPrompt('Can I carry my pet dog in a 3AC coach?')">🐕 Pet dog in 3AC policy</button>
          <button class="chip-btn" onclick="quickPrompt('Plan a journey from Pune to Delhi with cheapest AC class')">🧭 Plan Pune to Delhi trip</button>
        </div>
      </div>
    </div>

    <!-- Live Audio Waveform Visualizer -->
    <div class="audio-wave-container" id="waveBox">
      <div class="wave-bar"></div>
      <div class="wave-bar"></div>
      <div class="wave-bar"></div>
      <div class="wave-bar"></div>
      <div class="wave-bar"></div>
    </div>

    <!-- Composer Card -->
    <div class="composer-card">
      <div class="composer">
        <textarea id="input" placeholder="Type your message, or click 🎙️ Mic to talk directly…" rows="1"></textarea>
        <!-- The Prominent Microphone Button -->
        <button class="btn-mic-prominent" id="micBtn" title="Click to talk (live voice transcription)">🎙️</button>
        <!-- Send Button -->
        <button class="btn-send-prominent" id="sendBtn" disabled>
          <span>Send</span>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
        </button>
      </div>

      <div class="composer-footer">
        <div class="status-text" id="status">
          <span>Ready for your query (Press Enter ↵ to send)</span>
        </div>
        <button class="btn-clear" id="clearBtn">Clear Conversation</button>
      </div>
    </div>
  </div>
</main>

<script>
  // Robust API base resolving to local FastAPI server
  const API_BASE = window.location.origin.startsWith("http") ? window.location.origin : "http://127.0.0.1:8001";

  const log = document.getElementById("log");
  const welcomeScreen = document.getElementById("welcomeScreen");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("sendBtn");
  const micBtn = document.getElementById("micBtn");
  const waveBox = document.getElementById("waveBox");
  const statusBox = document.getElementById("status");
  const voiceToggle = document.getElementById("voiceToggle");
  const langSelect = document.getElementById("langSelect");
  const clearBtn = document.getElementById("clearBtn");

  let conversationHistory = [];
  let isRecording = false;
  let recognition = null;
  let activeAudio = null;

  // Auto-resize textarea
  function autoResize() {
    inputEl.style.height = "auto";
    const newH = Math.min(Math.max(inputEl.scrollHeight, 24), 140);
    inputEl.style.height = newH + "px";
    sendBtn.disabled = !inputEl.value.trim();
  }

  inputEl.addEventListener("input", autoResize);

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled) {
        sendMessage();
      }
    }
  });

  function quickPrompt(text) {
    inputEl.value = text;
    autoResize();
    sendMessage();
  }

  function setStatus(text, isPulse = false) {
    statusBox.innerHTML = `${isPulse ? '<span class="status-pulse"></span>' : ''}<span>${text}</span>`;
  }

  function showWave(show) {
    waveBox.style.display = show ? "flex" : "none";
  }

  // -------------------------------------------------------------------------
  // Speech Recognition (Real-Time Voice Dictation)
  // -------------------------------------------------------------------------
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  function initSpeech() {
    if (!SpeechRecognition) {
      alert("Microphone recognition is supported in Chrome, Edge, and Safari.");
      return null;
    }
    const sr = new SpeechRecognition();
    sr.continuous = true;
    sr.interimResults = true;

    sr.onstart = () => {
      isRecording = true;
      micBtn.classList.add("recording");
      showWave(true);
      setStatus("🎙️ Listening... speak into your microphone", true);
    };

    sr.onresult = (event) => {
      let fullTranscript = "";
      for (let i = 0; i < event.results.length; i++) {
        fullTranscript += event.results[i][0].transcript + " ";
      }
      inputEl.value = fullTranscript.trim();
      autoResize();
    };

    sr.onerror = (e) => {
      console.warn("Speech recognition error:", e);
      stopSpeech();
    };

    sr.onend = () => {
      stopSpeech();
    };

    return sr;
  }

  function toggleSpeech() {
    if (isRecording) {
      stopSpeech();
    } else {
      startSpeech();
    }
  }

  function startSpeech() {
    stopAudio();
    if (!recognition) {
      recognition = initSpeech();
    }
    if (!recognition) return;

    const lang = langSelect.value;
    if (lang === "hi") recognition.lang = "hi-IN";
    else if (lang === "ta") recognition.lang = "ta-IN";
    else if (lang === "te") recognition.lang = "te-IN";
    else if (lang === "bn") recognition.lang = "bn-IN";
    else if (lang === "mr") recognition.lang = "mr-IN";
    else recognition.lang = "en-IN";

    try {
      recognition.start();
    } catch (e) {
      console.warn(e);
    }
  }

  function stopSpeech() {
    isRecording = false;
    micBtn.classList.remove("recording");
    showWave(false);
    setStatus("Ready for your query (Press Enter ↵ to send)");
    if (recognition) {
      try { recognition.stop(); } catch (e) {}
    }
  }

  micBtn.onclick = toggleSpeech;

  // -------------------------------------------------------------------------
  // Messaging & Sarvam AI Integration
  // -------------------------------------------------------------------------
  async function sendMessage() {
    const query = inputEl.value.trim();
    if (!query) return;

    stopSpeech();
    stopAudio();

    if (welcomeScreen) welcomeScreen.style.display = "none";

    // Append User Message
    addMessage("me", query);
    conversationHistory.push({ role: "user", content: query });

    inputEl.value = "";
    autoResize();
    sendBtn.disabled = true;

    setStatus("⚡ RailDrishti AI is thinking…", true);

    const thinkingEl = addThinking();

    try {
      const resp = await fetch(API_BASE + "/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          language: langSelect.value || null,
          history: conversationHistory
        })
      });

      thinkingEl.remove();

      if (!resp.ok) {
        throw new Error("HTTP " + resp.status);
      }

      const data = await resp.json();
      const answer = data.answer || "No response received.";
      conversationHistory.push({ role: "assistant", content: answer });

      addMessage("bot", answer, data);
      setStatus("Ready for your query (Press Enter ↵ to send)");

      // Automatic Voice Playback (if Voice Toggle is ON)
      if (voiceToggle.checked) {
        speakResponse(answer);
      }

    } catch (err) {
      thinkingEl.remove();
      addMessage("bot", "⚠️ Could not connect to RailDrishti service (" + err.message + "). Please verify the server is running on port 8001.");
      setStatus("Connection error");
    }
  }

  sendBtn.onclick = sendMessage;

  function addThinking() {
    const el = document.createElement("div");
    el.className = "msg bot";
    el.innerHTML = `
      <div class="avatar">IR</div>
      <div class="bubble-wrap">
        <div class="bubble-author">RailDrishti AI</div>
        <div class="bubble" style="font-style:italic;color:var(--muted);display:flex;align-items:center;gap:8px;">
          <span>Thinking…</span>
          <div class="wave-bar" style="height:14px;background:var(--ir-navy)"></div>
          <div class="wave-bar" style="height:20px;background:var(--ir-navy)"></div>
          <div class="wave-bar" style="height:12px;background:var(--ir-navy)"></div>
        </div>
      </div>
    `;
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  }

  function addMessage(who, text, meta) {
    const el = document.createElement("div");
    el.className = "msg " + who;

    const author = who === "me" ? "You" : "RailDrishti AI";
    const initials = who === "me" ? "👤" : "IR";
    const formatted = who === "bot" ? renderMarkdown(text) : escapeHtml(text);

    let actions = "";
    if (who === "bot") {
      actions = `
        <div class="bubble-actions">
          <button class="action-btn" onclick="speakResponse(decodeURIComponent('${encodeURIComponent(text)}'), this)">
            🔊 Speak Voice
          </button>
          <button class="action-btn" onclick="copyAnswer(this, decodeURIComponent('${encodeURIComponent(text)}'))">
            📋 Copy
          </button>
          ${meta && meta.latency_ms ? `<span style="font-size:11px;color:var(--muted);margin-left:4px;">⚡ ${meta.model || 'Sarvam 105B'} (${meta.latency_ms}ms)</span>` : ''}
        </div>
      `;
    }

    el.innerHTML = `
      ${who === "bot" ? `<div class="avatar">${initials}</div>` : ''}
      <div class="bubble-wrap">
        <div class="bubble-author">${author}</div>
        <div class="bubble">${formatted}</div>
        ${actions}
      </div>
      ${who === "me" ? `<div class="avatar">${initials}</div>` : ''}
    `;

    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
  }

  // -------------------------------------------------------------------------
  // Voice Output (Sarvam Bulbul TTS + Web Speech)
  // -------------------------------------------------------------------------
  function stopAudio() {
    showWave(false);
    if (activeAudio) {
      activeAudio.pause();
      activeAudio = null;
    }
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    document.querySelectorAll(".action-btn.active-speak").forEach(b => {
      b.classList.remove("active-speak");
      b.innerHTML = "🔊 Speak Voice";
    });
  }

  async function speakResponse(text, btn = null) {
    stopAudio();

    if (btn) {
      btn.classList.add("active-speak");
      btn.innerHTML = "⏹️ Stop";
    }

    showWave(true);

    try {
      // Call Sarvam Bulbul TTS
      const resp = await fetch(API_BASE + "/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          language: langSelect.value || "en"
        })
      });

      if (resp.ok) {
        const data = await resp.json();
        if (data.audio_url) {
          activeAudio = new Audio(data.audio_url);
          activeAudio.onended = () => {
            stopAudio();
          };
          activeAudio.onerror = () => {
            browserSpeak(text);
          };
          activeAudio.play();
          return;
        }
      }
    } catch (e) {
      console.warn("Sarvam TTS error, falling back to browser speech:", e);
    }

    browserSpeak(text);
  }

  function browserSpeak(text) {
    if (!("speechSynthesis" in window)) {
      showWave(false);
      return;
    }
    const clean = text.replace(/[*_#`\\[\\]]/g, "").slice(0, 500);
    const u = new SpeechSynthesisUtterance(clean);
    u.lang = langSelect.value === "hi" ? "hi-IN" : "en-IN";
    u.rate = 1.0;
    u.onend = () => stopAudio();
    u.onerror = () => stopAudio();
    window.speechSynthesis.speak(u);
  }

  function copyAnswer(btn, text) {
    navigator.clipboard.writeText(text).then(() => {
      const orig = btn.innerHTML;
      btn.innerHTML = "✓ Copied!";
      setTimeout(() => { btn.innerHTML = orig; }, 1500);
    });
  }

  clearBtn.onclick = () => {
    stopAudio();
    stopSpeech();
    conversationHistory = [];
    log.innerHTML = "";
    if (welcomeScreen) {
      log.appendChild(welcomeScreen);
      welcomeScreen.style.display = "block";
    }
  };

  function escapeHtml(t) {
    const d = document.createElement("div");
    d.innerText = t;
    return d.innerHTML;
  }

  function renderMarkdown(md) {
    if (!md) return "";
    let html = escapeHtml(md);
    html = html.replace(/```([\\s\\S]*?)```/g, '<pre style="background:#F1F5F9;padding:8px;border-radius:8px;overflow-x:auto"><code>$1</code></pre>');
    html = html.replace(/`([^`]+)`/g, '<code style="background:#F1F5F9;padding:2px 6px;border-radius:4px">$1</code>');
    html = html.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
    html = html.replace(/\\*(.*?)\\*/g, '<em>$1</em>');
    html = html.replace(/^### (.*$)/gim, '<h4 style="color:var(--ir-navy);margin:10px 0 4px">$1</h4>');
    html = html.replace(/^## (.*$)/gim, '<h3 style="color:var(--ir-navy);margin:12px 0 6px">$1</h3>');
    html = html.replace(/^# (.*$)/gim, '<h2 style="color:var(--ir-navy);margin:14px 0 8px">$1</h2>');
    html = html.replace(/^\\* (.*$)/gim, '<li>$1</li>');
    html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\\/li>)/s, '<ul>$1</ul>');
    html = html.replace(/\\n\\n/g, '<p></p>');
    html = html.replace(/\\n/g, '<br>');
    return html;
  }
</script>
</body>
</html>
"""
