const el = (sel) => document.querySelector(sel);

function appendMessage(role, text) {
  const wrap = el('#messages');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = text;
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
  return div;
}

el('#uploadForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = el('#file').files[0];
  if (!file) return;
  el('#uploadStatus').textContent = 'Subiendo e indexando...';
  const form = new FormData();
  form.append('file', file);
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: form });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    el('#uploadStatus').textContent = `OK: ${data.total_paragraphs} párrafos`;
  } catch (err) {
    el('#uploadStatus').textContent = `Error: ${err}`;
  }
});

el('#chatForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = el('#prompt');
  const prompt = input.value.trim();
  const mode = el('#rankMode').value;
  const embedMode = el('#embedMode').value;
  if (!prompt) return;
  appendMessage('user', prompt);
  const isRerankModelo = (mode === 'modelo');
  let thinkingDiv = null;
  if (isRerankModelo) {
    thinkingDiv = appendMessage('bot', '⏳ Re‑rankeando con modelo...');
  }
  input.value = '';
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, mode, embedMode })
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    if (thinkingDiv) {
      thinkingDiv.textContent = data.answer || '(sin respuesta)';
    } else {
      appendMessage('bot', data.answer || '(sin respuesta)');
    }
  } catch (err) {
    if (thinkingDiv) {
      thinkingDiv.textContent = `Error: ${err}`;
    } else {
      appendMessage('bot', `Error: ${err}`);
    }
  }
});


