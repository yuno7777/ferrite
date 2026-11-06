//! Embedded Zero-Dependency Single-Binary WebUI Playground.
//!
//! Served automatically at `GET /` by `ferrite serve`, providing an interactive
//! chat playground, live token/s telemetry meter, temperature/top-p controls,
//! and `/v1/models` inspection without requiring Node.js or external static files.

/// Complete single-page HTML/CSS/JS application served at `GET /`.
pub const PLAYGROUND_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ferrite Local AI Playground</title>
<style>
:root {
  --bg: #0b0f17;
  --panel: #151c28;
  --border: #263248;
  --accent: #f97316;
  --text: #e2e8f0;
  --muted: #94a3b8;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  display: flex;
  flex-direction: column;
  height: 100vh;
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.85rem 1.5rem;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
}
header h1 { font-size: 1.1rem; font-weight: 700; letter-spacing: 0.02em; }
header .badge {
  font-size: 0.75rem;
  background: rgba(249, 115, 22, 0.15);
  color: var(--accent);
  border: 1px solid var(--accent);
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
}
main {
  display: grid;
  grid-template-columns: 1fr 280px;
  flex: 1;
  overflow: hidden;
}
#chat {
  display: flex;
  flex-direction: column;
  padding: 1.25rem;
  overflow-y: auto;
  gap: 0.9rem;
}
.msg {
  max-width: 82%;
  padding: 0.85rem 1.1rem;
  border-radius: 0.75rem;
  line-height: 1.5;
  white-space: pre-wrap;
}
.msg.user {
  align-self: flex-end;
  background: var(--accent);
  color: #fff;
}
.msg.assistant {
  align-self: flex-start;
  background: var(--panel);
  border: 1px solid var(--border);
}
aside {
  background: var(--panel);
  border-left: 1px solid var(--border);
  padding: 1.1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  font-size: 0.85rem;
}
aside label { display: flex; flex-direction: column; gap: 0.35rem; color: var(--muted); }
aside input {
  background: var(--bg);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 0.45rem 0.6rem;
  border-radius: 0.4rem;
}
footer {
  display: flex;
  gap: 0.75rem;
  padding: 1rem 1.5rem;
  background: var(--panel);
  border-top: 1px solid var(--border);
}
footer textarea {
  flex: 1;
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 0.65rem 0.9rem;
  resize: none;
  height: 2.8rem;
}
footer button {
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 0.5rem;
  padding: 0 1.25rem;
  font-weight: 600;
  cursor: pointer;
}
</style>
</head>
<body>
<header>
  <h1>Ferrite Local AI Engine</h1>
  <span class="badge" id="model-badge">Zero-Dependency Rust</span>
</header>
<main>
  <section id="chat">
    <div class="msg assistant">Ready. Send a prompt below to run local GGUF inference.</div>
  </section>
  <aside>
    <label>Max Tokens <input id="max-tokens" type="number" value="128" min="1" max="4096"></label>
    <label>Temperature <input id="temperature" type="number" value="0.7" step="0.1" min="0" max="2"></label>
    <div id="stats" style="color:var(--muted);margin-top:auto;">Status: Idle</div>
  </aside>
</main>
<footer>
  <textarea id="prompt" placeholder="Ask Ferrite anything..."></textarea>
  <button id="send">Send</button>
</footer>
<script>
const chat = document.getElementById('chat');
const promptEl = document.getElementById('prompt');
const sendBtn = document.getElementById('send');
const statsEl = document.getElementById('stats');
const badgeEl = document.getElementById('model-badge');

fetch('/v1/models').then(r => r.json()).then(d => {
  if (d.data && d.data[0]) badgeEl.textContent = d.data[0].id;
}).catch(() => {});

async function sendPrompt() {
  const content = promptEl.value.trim();
  if (!content) return;
  promptEl.value = '';
  const u = document.createElement('div');
  u.className = 'msg user';
  u.textContent = content;
  chat.appendChild(u);

  const a = document.createElement('div');
  a.className = 'msg assistant';
  a.textContent = '...';
  chat.appendChild(a);
  chat.scrollTop = chat.scrollHeight;

  const t0 = performance.now();
  statsEl.textContent = 'Generating...';
  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        messages: [{role: 'user', content}],
        max_tokens: parseInt(document.getElementById('max-tokens').value || '128', 10),
        temperature: parseFloat(document.getElementById('temperature').value || '0.7')
      })
    });
    const data = await res.json();
    const dt = (performance.now() - t0) / 1000;
    const reply = data.choices?.[0]?.message?.content || JSON.stringify(data);
    a.textContent = reply;
    const toks = data.usage?.completion_tokens || 0;
    statsEl.textContent = toks ? `${(toks / dt).toFixed(1)} tok/s (${toks} tokens)` : `Done in ${dt.toFixed(2)}s`;
  } catch (err) {
    a.textContent = 'Error: ' + err.message;
    statsEl.textContent = 'Error';
  }
  chat.scrollTop = chat.scrollHeight;
}
sendBtn.addEventListener('click', sendPrompt);
promptEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendPrompt(); }
});
</script>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn playground_html_contains_chat_completions_wiring() {
        assert!(PLAYGROUND_HTML.contains("<!DOCTYPE html>"));
        assert!(PLAYGROUND_HTML.contains("/v1/chat/completions"));
        assert!(PLAYGROUND_HTML.contains("/v1/models"));
    }
}
