const el = (sel) => document.querySelector(sel);

function appendMessage(role, text) {
  const wrap = el('#messages');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = text;
  wrap.appendChild(div);
  wrap.scrollTop = wrap.scrollHeight;
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
  if (!prompt) return;
  appendMessage('user', prompt);
  input.value = '';
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    appendMessage('bot', data.answer || '(sin respuesta)');
  } catch (err) {
    appendMessage('bot', `Error: ${err}`);
  }
});


