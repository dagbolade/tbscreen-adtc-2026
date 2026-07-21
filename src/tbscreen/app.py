#!/usr/bin/env python3
"""Local Flask clinical dashboard — image screening + grounded Q&A, 100% offline."""

from __future__ import annotations

import logging
import os
import sys
import traceback
import uuid
from logging.handlers import RotatingFileHandler
from typing import Any

from flask import Flask, jsonify, request, render_template_string
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tbscreen import TBScreenAssistant  # noqa: E402

LOG_DIR = os.path.join(_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def _setup_logging() -> logging.Logger:
    """Write Flask + app logs to logs/tbscreen.log (and stderr)."""
    logger = logging.getLogger("tbscreen")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, "tbscreen.log"),
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    # Capture werkzeug access + Flask request logs in the same file.
    for name in ("werkzeug", "tbscreen.llm", "tbscreen.app"):
        w = logging.getLogger(name)
        w.setLevel(logging.INFO)
        w.addHandler(fh)
        w.propagate = False
        w.addHandler(sh)
    return logger


log = _setup_logging()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(_ROOT, "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXT = {".png", ".jpg", ".jpeg"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

assistant: TBScreenAssistant | None = None
_assistant_lock = __import__("threading").Lock()


def get_assistant() -> TBScreenAssistant:
    """Lazy-init assistant so cold start stays light until first request."""
    global assistant
    with _assistant_lock:
        if assistant is None:
            log.info("Initializing TBScreenAssistant (ONNX + RAG + GGUF)")
            assistant = TBScreenAssistant()
        return assistant


def _parse_patient_context(form_or_json: Any) -> dict[str, Any]:
    """Collect only the minimal fields required by the WHO-aligned safety policy."""
    src = form_or_json or {}
    ctx: dict[str, Any] = {}

    def _num(key: str) -> float | None:
        raw = src.get(key)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _bool(key: str) -> bool | None:
        raw = src.get(key)
        if raw is None or raw == "":
            return None
        if isinstance(raw, bool):
            return raw
        return str(raw).lower() in {"1", "true", "yes", "on"}

    age = _num("age_years")
    if age is not None:
        ctx["age_years"] = age
    cough = _num("cough_weeks")
    if cough is not None:
        ctx["cough_weeks"] = cough
    for key in ("has_tb_symptoms", "hiv_positive", "household_contact"):
        val = _bool(key)
        if val is not None:
            ctx[key] = val
    return ctx


def _validate_image(path: str) -> None:
    """Reject non-image uploads before ONNX inference."""
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        img.load()


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>TBScreen — Offline Clinical TB Assistant</title>
  <style>
    :root {
      --bg:#080B11; --card:rgba(17,22,34,.75); --border:rgba(255,255,255,.08);
      --text:#F3F4F6; --muted:#9CA3AF; --blue:#3B82F6; --green:#10B981;
      --amber:#F59E0B; --red:#EF4444;
    }
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:"Segoe UI","Helvetica Neue",sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
    header{display:flex;justify-content:space-between;align-items:center;padding:1.25rem 1.5rem;border-bottom:1px solid var(--border)}
    .logo{font-weight:700;font-size:1.35rem;background:linear-gradient(135deg,#3B82F6,#10B981);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .badge{border:1px solid rgba(16,185,129,.35);color:var(--green);padding:.25rem .7rem;border-radius:999px;font-size:.8rem}
    main{max-width:1200px;margin:0 auto;padding:1.5rem;display:grid;gap:1.25rem}
    @media(min-width:960px){main{grid-template-columns:380px 1fr}}
    .card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.25rem}
    h2{font-size:1.1rem;margin-bottom:.85rem}
    label{display:block;font-size:.8rem;color:var(--muted);margin:.55rem 0 .25rem}
    input,textarea,select{width:100%;background:#0d121c;border:1px solid var(--border);color:var(--text);border-radius:8px;padding:.55rem .7rem}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:.6rem}
    .dropzone{border:2px dashed rgba(255,255,255,.15);border-radius:12px;padding:1.5rem;text-align:center;cursor:pointer;color:var(--muted)}
    .dropzone:hover{border-color:var(--blue)}
    #preview{display:none;width:100%;max-height:240px;object-fit:contain;margin-top:.75rem;border-radius:8px;background:#000}
    .btn{width:100%;margin-top:.85rem;border:none;border-radius:10px;padding:.8rem;font-weight:600;cursor:pointer;color:#fff;background:linear-gradient(135deg,#3B82F6,#1D4ED8)}
    .btn:disabled{opacity:.5;cursor:not-allowed}
    .btn.secondary{background:#1f2937;border:1px solid var(--border)}
    .tabs{display:flex;gap:.4rem;margin-bottom:1rem}
    .tab{background:#111827;border:1px solid var(--border);color:var(--muted);padding:.4rem .75rem;border-radius:8px;cursor:pointer}
    .tab.active{background:var(--blue);color:#fff;border-color:var(--blue)}
    .panel{display:none}.panel.active{display:block}
    .lang-bar{display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:1rem}
    .btn-lang{background:#111827;border:1px solid var(--border);color:var(--muted);padding:.35rem .7rem;border-radius:8px;cursor:pointer}
    .btn-lang.active{background:var(--blue);color:#fff}
    .metric{font-size:2rem;font-weight:700}
    .muted{color:var(--muted);font-size:.85rem}
    .box{margin-top:.85rem;padding:.85rem;border-radius:10px;border:1px solid var(--border);background:rgba(0,0,0,.2)}
    .caution{color:#F87171;font-size:.85rem;margin-top:.35rem}
    .source{display:inline-block;background:rgba(59,130,246,.12);color:#93C5FD;border:1px solid rgba(59,130,246,.25);border-radius:4px;padding:0 .35rem;font-size:.75rem;margin-left:.25rem}
    footer{text-align:center;color:var(--muted);font-size:.75rem;padding:1rem;border-top:1px solid var(--border)}
    .loader{display:none;text-align:center;padding:2rem;color:var(--muted)}
  </style>
</head>
<body>
<header>
  <div class="logo">TBScreen</div>
  <div class="badge">Offline</div>
</header>
<main>
  <section class="card">
    <div class="tabs">
      <button class="tab active" data-panel="screen">CXR Screen</button>
      <button class="tab" data-panel="qa">Clinical Q&amp;A</button>
    </div>

    <div id="panel-screen" class="panel active">
      <h2>Chest X-ray screening</h2>
      <div class="dropzone" id="dropzone">Drop PNG/JPEG CXR or click to browse</div>
      <input type="file" id="file-input" accept="image/png,image/jpeg" hidden/>
      <img id="preview" alt="CXR preview"/>
      <div class="row">
        <div><label>Age (years)</label><input id="age_years" type="number" min="0" max="120" placeholder="e.g. 34"/></div>
        <div><label>Cough (weeks)</label><input id="cough_weeks" type="number" min="0" max="52" placeholder="e.g. 3"/></div>
      </div>
      <label><input id="has_tb_symptoms" type="checkbox"/> TB symptoms present</label>
      <label><input id="hiv_positive" type="checkbox"/> Living with HIV</label>
      <label><input id="household_contact" type="checkbox"/> Household TB contact</label>
      <button class="btn" id="btn-analyze" disabled>Screen &amp; interpret</button>
    </div>

    <div id="panel-qa" class="panel">
      <h2>Guideline Q&amp;A</h2>
      <label>Question</label>
      <textarea id="qa-question" rows="5" placeholder="Ask a WHO-guideline grounded clinical question…"></textarea>
      <button class="btn secondary" id="btn-ask">Ask (offline RAG)</button>
    </div>
  </section>

  <section class="card">
    <div class="lang-bar">
      <button class="btn-lang active" data-lang="English">English</button>
      <button class="btn-lang" data-lang="Yoruba">Yorùbá</button>
      <button class="btn-lang" data-lang="Hausa">Hausa</button>
      <button class="btn-lang" data-lang="Igbo">Igbo</button>
    </div>
    <div class="loader" id="loader">Running offline ONNX + RAG + GGUF…</div>
    <div id="empty" class="muted">Upload a CXR or ask a clinical question. Results never persist across sessions.</div>
    <div id="results" style="display:none">
      <div class="metric" id="prob">—</div>
      <div class="muted" id="triage">—</div>
      <div class="box"><strong>Interpretation / Answer</strong><div id="body-text"></div></div>
      <div class="box"><strong>Recommendation</strong><div id="rec-text"></div></div>
      <div class="box"><strong>Patient education</strong><ul id="edu-list"></ul></div>
      <div class="box" id="cautions"></div>
    </div>
  </section>
</main>
<footer>TBScreen • MobileNetV3-ONNX + local GGUF • decision support only, not diagnosis</footer>
<script>
  let selectedFile=null, currentLang="English", mode="screen";
  const $ = (id)=>document.getElementById(id);
  const dropzone=$("dropzone"), fileInput=$("file-input"), preview=$("preview");
  const btnAnalyze=$("btn-analyze"), btnAsk=$("btn-ask"), loader=$("loader");
  const empty=$("empty"), results=$("results");

  document.querySelectorAll(".tab").forEach(btn=>{
    btn.onclick=()=>{
      document.querySelectorAll(".tab").forEach(b=>b.classList.remove("active"));
      btn.classList.add("active");
      mode=btn.dataset.panel;
      document.querySelectorAll(".panel").forEach(p=>p.classList.remove("active"));
      $("panel-"+mode).classList.add("active");
    };
  });
  document.querySelectorAll(".btn-lang").forEach(btn=>{
    btn.onclick=()=>{
      document.querySelectorAll(".btn-lang").forEach(b=>b.classList.remove("active"));
      btn.classList.add("active");
      currentLang=btn.dataset.lang;
      if(mode==="screen" && results.style.display!=="none"){
        postJSON("/translate",{lang:currentLang}).then(renderScreen).catch(()=>{});
      }
    };
  });

  dropzone.onclick=()=>fileInput.click();
  dropzone.ondragover=e=>{e.preventDefault();};
  dropzone.ondrop=e=>{e.preventDefault(); if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);};
  fileInput.onchange=e=>{ if(e.target.files[0]) handleFile(e.target.files[0]); };

  function handleFile(file){
    selectedFile=file;
    const reader=new FileReader();
    reader.onload=ev=>{ preview.src=ev.target.result; preview.style.display="block"; btnAnalyze.disabled=false; };
    reader.readAsDataURL(file);
  }

  function patientFields(){
    return {
      age_years: $("age_years").value,
      cough_weeks: $("cough_weeks").value,
      has_tb_symptoms: $("has_tb_symptoms").checked,
      hiv_positive: $("hiv_positive").checked,
      household_contact: $("household_contact").checked,
    };
  }

  function setLoading(on){
    loader.style.display=on?"block":"none";
    empty.style.display=on?"none":(results.style.display==="none"?"block":"none");
  }

  function escapeHtml(s){
    return String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
  }
  function withCitations(s){
    return escapeHtml(s).replace(/\[([a-zA-Z0-9\-]+)\]/g,'<span class="source">$1</span>');
  }

  function renderScreen(data){
    if(data.error){ alert(data.error); return; }
    empty.style.display="none"; results.style.display="block";
    const prob=Math.round((data.vision_result?.tb_probability||0)*100);
    $("prob").textContent=prob+"% TB probability";
    $("triage").textContent=`Triage: ${(data.triage||"—").toUpperCase()} | Risk: ${data.risk_level||"—"}`;
    const interp=data.interpretation||{};
    $("body-text").innerHTML=withCitations(interp.interpretation||"");
    $("rec-text").innerHTML=withCitations(interp.recommendation||"");
    $("edu-list").innerHTML=(interp.education||[]).map(p=>`<li>${withCitations(p)}</li>`).join("");
    $("cautions").innerHTML="<strong>Cautions</strong>"+(interp.cautions||[]).map(c=>`<div class="caution">${escapeHtml(c)}</div>`).join("");
  }

  function renderQA(data){
    if(data.error){ alert(data.error); return; }
    empty.style.display="none"; results.style.display="block";
    $("prob").textContent="Clinical Q&A";
    $("triage").textContent="Grounded in offline WHO passages";
    const a=data.answer||{};
    $("body-text").innerHTML=withCitations(a.answer||"");
    $("rec-text").innerHTML=withCitations(a.recommendation||"");
    $("edu-list").innerHTML=(a.education||[]).map(p=>`<li>${withCitations(p)}</li>`).join("");
    $("cautions").innerHTML="<strong>Cautions</strong>"+(a.cautions||[]).map(c=>`<div class="caution">${escapeHtml(c)}</div>`).join("");
  }

  async function postJSON(url, body){
    const res=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    return res.json();
  }

  btnAnalyze.onclick=async()=>{
    if(!selectedFile) return;
    setLoading(true); results.style.display="none"; btnAnalyze.disabled=true;
    const fd=new FormData();
    fd.append("file", selectedFile);
    fd.append("lang", currentLang);
    const ctx=patientFields();
    Object.entries(ctx).forEach(([k,v])=>fd.append(k, v));
    try{
      const res=await fetch("/analyze",{method:"POST",body:fd});
      const data=await res.json().catch(()=>({error:"Bad JSON from server (process may have crashed — check logs/tbscreen.log)"}));
      setLoading(false); btnAnalyze.disabled=false;
      if(!res.ok || data.error){ alert(data.error || ("HTTP "+res.status)); empty.style.display="block"; return; }
      renderScreen(data);
    }catch(e){
      setLoading(false); btnAnalyze.disabled=false;
      alert("Request failed: "+e+". If the terminal shows GGML_ASSERT / abort, restart the app and check logs/tbscreen.log");
      empty.style.display="block";
    }
  };

  btnAsk.onclick=async()=>{
    const question=$("qa-question").value.trim();
    if(!question){ alert("Enter a question"); return; }
    setLoading(true); results.style.display="none";
    try{
      const res=await fetch("/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question, lang:currentLang})});
      const data=await res.json().catch(()=>({error:"Bad JSON from server (process may have crashed — check logs/tbscreen.log)"}));
      setLoading(false);
      if(!res.ok || data.error){ alert(data.error || ("HTTP "+res.status)); empty.style.display="block"; return; }
      renderQA(data);
    }catch(e){
      setLoading(false);
      alert("Request failed: "+e+". Server may have crashed — check logs/tbscreen.log");
      empty.style.display="block";
    }
  };
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "Only PNG/JPEG uploads are allowed"}), 400

    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    lang = request.form.get("lang", "English")
    patient_context = _parse_patient_context(request.form)
    log.info("POST /analyze lang=%s patient=%s file=%s", lang, patient_context, filename)

    try:
        _validate_image(filepath)
        assist = get_assistant()
        assist.clear_session()
        result = assist.process_image(filepath, lang=lang, patient_context=patient_context or None)
        log.info(
            "POST /analyze ok triage=%s risk=%s sources=%s",
            result.get("triage"),
            result.get("risk_level"),
            result.get("retrieved_sources"),
        )
        return jsonify(result)
    except Exception as e:  # noqa: BLE001 — surface clean API error to UI
        log.error("POST /analyze failed: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass


@app.route("/translate", methods=["POST"])
def translate():
    """Re-interpret cached vision result in a new language — skips ONNX."""
    data = request.get_json() or {}
    lang = data.get("lang", "English")
    log.info("POST /translate lang=%s", lang)
    try:
        assist = get_assistant()
        result = assist.reinterpret(lang=lang)
        return jsonify(result)
    except ValueError as e:
        log.warning("POST /translate bad request: %s", e)
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        log.error("POST /translate failed: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """Grounded clinical Q&A (matches metadata.json test-prompt style)."""
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    lang = data.get("lang", "English")
    if not question:
        return jsonify({"error": "question is required"}), 400
    log.info("POST /ask lang=%s q_chars=%s", lang, len(question))
    try:
        assist = get_assistant()
        result = assist.ask(question, lang=lang)
        log.info("POST /ask ok sources=%s", result.get("retrieved_sources"))
        return jsonify(result)
    except Exception as e:  # noqa: BLE001
        log.error("POST /ask failed: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/session/clear", methods=["POST"])
def clear_session():
    """Explicitly drop cached vision/patient state between patients."""
    if assistant is not None:
        assistant.clear_session()
    return jsonify({"ok": True})


if __name__ == "__main__":
    host = os.environ.get("TBSCREEN_HOST", "127.0.0.1")
    port = int(os.environ.get("TBSCREEN_PORT", "5000"))
    log.info("Starting TBScreen on http://%s:%s (logs → %s)", host, port, LOG_DIR)
    print(f"Starting TBScreen on http://{host}:{port}")
    print(f"Logs: {os.path.join(LOG_DIR, 'tbscreen.log')}")
    print(f"Launch alternatives:  PYTHONPATH=src:. python src/tbscreen/app.py")
    # threaded=False: llama.cpp context must not serve concurrent requests.
    app.run(host=host, port=port, debug=False, threaded=False)
