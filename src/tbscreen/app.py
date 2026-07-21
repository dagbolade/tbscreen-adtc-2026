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
  <meta name="description" content="TBScreen: an offline clinical decision-support tool for TB chest X-ray screening and WHO-guideline Q&A."/>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"/>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    :root{
      --bg:#F8F9FB;--surface:#FFFFFF;--border:#E2E5EB;
      --text:#1A1D23;--text-secondary:#5F6672;--text-tertiary:#8B919D;
      --primary:#1570EF;--primary-hover:#1259C4;--primary-light:#EFF5FF;
      --green:#0D7C42;--green-bg:#ECFDF3;
      --amber:#B25E09;--amber-bg:#FFF7ED;
      --red:#C4320A;--red-bg:#FEF3F2;
      --radius:10px;--radius-sm:6px;
    }
    html{-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
    body{font-family:'Inter',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;font-size:14px;line-height:1.5}

    /* ── Header ─────────────────────────────────────── */
    header{
      background:var(--surface);border-bottom:1px solid var(--border);
      display:flex;justify-content:space-between;align-items:center;
      padding:.875rem 1.5rem;
    }
    .logo{font-weight:700;font-size:1.2rem;color:var(--text);letter-spacing:-.02em}
    .logo span{color:var(--primary)}
    .badge{
      display:inline-flex;align-items:center;gap:5px;
      color:var(--green);background:var(--green-bg);
      padding:.25rem .65rem;border-radius:999px;font-size:.75rem;font-weight:500;
    }
    .badge::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--green)}

    /* ── Layout ─────────────────────────────────────── */
    main{max-width:1120px;margin:0 auto;padding:1.25rem;display:grid;gap:1.25rem}
    @media(min-width:960px){main{grid-template-columns:380px 1fr}}

    /* ── Card ───────────────────────────────────────── */
    .card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.25rem}

    /* ── Tabs ───────────────────────────────────────── */
    .tabs{display:flex;gap:2px;background:var(--bg);border-radius:var(--radius-sm);padding:3px;margin-bottom:1.25rem}
    .tab{
      flex:1;text-align:center;background:transparent;border:none;
      color:var(--text-secondary);padding:.45rem .75rem;border-radius:5px;
      cursor:pointer;font-size:.8125rem;font-weight:500;font-family:inherit;
      transition:all .15s ease;
    }
    .tab:hover{color:var(--text)}
    .tab.active{background:var(--surface);color:var(--text);box-shadow:0 1px 3px rgba(0,0,0,.06)}
    .panel{display:none}.panel.active{display:block}

    /* ── Section heading ───────────────────────────── */
    h2{font-size:.9375rem;font-weight:600;margin-bottom:.85rem;color:var(--text)}

    /* ── Labels & inputs ───────────────────────────── */
    label.field-label{display:block;font-size:.75rem;font-weight:500;color:var(--text-secondary);margin-bottom:.3rem;margin-top:.85rem}
    input[type="number"],textarea{
      width:100%;background:var(--surface);
      border:1px solid var(--border);color:var(--text);
      border-radius:var(--radius-sm);padding:.5rem .65rem;
      font-family:inherit;font-size:.8125rem;
      transition:border-color .15s ease;
    }
    input[type="number"]:focus,textarea:focus{outline:none;border-color:var(--primary);box-shadow:0 0 0 3px rgba(21,112,239,.1)}
    textarea{resize:vertical}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:.75rem}

    /* ── Checkbox rows ─────────────────────────────── */
    .check-row{
      display:flex;align-items:center;gap:.5rem;
      margin-top:.65rem;cursor:pointer;user-select:none;
      font-size:.8125rem;color:var(--text-secondary);
    }
    .check-row input[type="checkbox"]{
      width:16px;height:16px;accent-color:var(--primary);cursor:pointer;
      border-radius:3px;flex-shrink:0;
    }

    /* ── Dropzone ───────────────────────────────────── */
    .dropzone{
      border:1.5px dashed var(--border);border-radius:var(--radius);
      padding:1.75rem 1rem;text-align:center;cursor:pointer;
      color:var(--text-tertiary);font-size:.8125rem;
      transition:border-color .15s ease, background .15s ease;
    }
    .dropzone:hover{border-color:var(--primary);background:var(--primary-light)}
    .dropzone-icon{font-size:1.5rem;margin-bottom:.35rem;display:block}
    #preview{display:none;width:100%;max-height:200px;object-fit:contain;margin-top:.75rem;border-radius:var(--radius-sm);background:var(--bg)}

    /* ── Buttons ────────────────────────────────────── */
    .btn{
      width:100%;margin-top:1rem;border:none;border-radius:var(--radius-sm);
      padding:.65rem;font-weight:600;cursor:pointer;font-family:inherit;
      font-size:.8125rem;color:#fff;background:var(--primary);
      transition:background .15s ease;
    }
    .btn:hover:not(:disabled){background:var(--primary-hover)}
    .btn:disabled{opacity:.45;cursor:not-allowed}
    .btn.secondary{background:var(--bg);border:1px solid var(--border);color:var(--text)}
    .btn.secondary:hover:not(:disabled){background:#ECEEF1}

    /* ── Language bar ───────────────────────────────── */
    .lang-bar{display:flex;gap:2px;background:var(--bg);border-radius:var(--radius-sm);padding:3px;margin-bottom:1.25rem}
    .btn-lang{
      background:transparent;border:none;color:var(--text-secondary);
      padding:.35rem .7rem;border-radius:5px;cursor:pointer;
      font-size:.8125rem;font-weight:500;font-family:inherit;
      transition:all .15s ease;
    }
    .btn-lang:hover{color:var(--text)}
    .btn-lang.active{background:var(--surface);color:var(--text);box-shadow:0 1px 3px rgba(0,0,0,.06)}

    /* ── Results ────────────────────────────────────── */
    .metric{font-size:1.75rem;font-weight:700;color:var(--text);letter-spacing:-.02em}
    .muted{color:var(--text-tertiary);font-size:.8125rem}
    .triage-line{color:var(--text-secondary);font-size:.8125rem;margin-top:.15rem}
    .result-section{margin-top:1rem;padding:.85rem;border-radius:var(--radius-sm);border:1px solid var(--border);background:var(--bg)}
    .result-section strong{font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.03em;color:var(--text-secondary);display:block;margin-bottom:.4rem}
    .result-section div,.result-section ul{font-size:.8125rem;color:var(--text);line-height:1.6}
    .result-section ul{padding-left:1.1rem}
    .result-section li{margin-bottom:.2rem}
    .caution{color:var(--red);font-size:.8125rem;margin-top:.3rem}
    .source{
      display:inline;background:var(--primary-light);color:var(--primary);
      border-radius:3px;padding:0 .3rem;font-size:.7rem;font-weight:500;
    }

    /* ── Empty state ───────────────────────────────── */
    .empty-state{text-align:center;padding:3rem 1rem;color:var(--text-tertiary)}
    .empty-state .icon{font-size:2rem;margin-bottom:.5rem;display:block;opacity:.5}

    /* ── Loader ─────────────────────────────────────── */
    .loader{display:none;text-align:center;padding:2.5rem 1rem;color:var(--text-tertiary)}
    .loader .spinner{
      width:24px;height:24px;border:2.5px solid var(--border);border-top-color:var(--primary);
      border-radius:50%;margin:0 auto .75rem;
      animation:spin .7s linear infinite;
    }
    @keyframes spin{to{transform:rotate(360deg)}}

    /* ── Footer ─────────────────────────────────────── */
    footer{text-align:center;color:var(--text-tertiary);font-size:.7rem;padding:1.25rem;border-top:1px solid var(--border);margin-top:.5rem}
  </style>
</head>
<body>
<header>
  <div class="logo">TB<span>Screen</span></div>
  <div class="badge">Offline</div>
</header>
<main>
  <!-- ─── Left panel ─── -->
  <section class="card">
    <div class="tabs">
      <button class="tab active" data-panel="screen">CXR Screen</button>
      <button class="tab" data-panel="qa">Clinical Q&amp;A</button>
    </div>

    <div id="panel-screen" class="panel active">
      <h2>Chest X-ray screening</h2>
      <div class="dropzone" id="dropzone">
        <span class="dropzone-icon">📁</span>
        Drop a PNG / JPEG chest X-ray here, or click to browse
      </div>
      <input type="file" id="file-input" accept="image/png,image/jpeg" hidden/>
      <img id="preview" alt="CXR preview"/>
      <div class="row">
        <div><label class="field-label">Age (years)</label><input id="age_years" type="number" min="0" max="120" placeholder="e.g. 34"/></div>
        <div><label class="field-label">Cough (weeks)</label><input id="cough_weeks" type="number" min="0" max="52" placeholder="e.g. 3"/></div>
      </div>
      <label class="check-row"><input id="has_tb_symptoms" type="checkbox"/> TB symptoms present</label>
      <label class="check-row"><input id="hiv_positive" type="checkbox"/> Living with HIV</label>
      <label class="check-row"><input id="household_contact" type="checkbox"/> Household TB contact</label>
      <button class="btn" id="btn-analyze" disabled>Screen &amp; Interpret</button>
    </div>

    <div id="panel-qa" class="panel">
      <h2>Guideline Q&amp;A</h2>
      <label class="field-label">Question</label>
      <textarea id="qa-question" rows="5" placeholder="Ask a WHO-guideline grounded clinical question…"></textarea>
      <button class="btn secondary" id="btn-ask">Ask (offline RAG)</button>
    </div>
  </section>

  <!-- ─── Right panel ─── -->
  <section class="card">
    <div class="lang-bar">
      <button class="btn-lang active" data-lang="English">English</button>
      <button class="btn-lang" data-lang="Yoruba">Yorùbá</button>
      <button class="btn-lang" data-lang="Hausa">Hausa</button>
      <button class="btn-lang" data-lang="Igbo">Igbo</button>
    </div>
    <div class="loader" id="loader">
      <div class="spinner"></div>
      Running offline ONNX + RAG + GGUF…
    </div>
    <div id="empty" class="empty-state">
      <span class="icon">🩻</span>
      Upload a CXR or ask a clinical question.<br/>Results never persist across sessions.
    </div>
    <div id="results" style="display:none">
      <div class="metric" id="prob">—</div>
      <div class="triage-line" id="triage">—</div>
      <div class="result-section"><strong>Interpretation / Answer</strong><div id="body-text"></div></div>
      <div class="result-section"><strong>Recommendation</strong><div id="rec-text"></div></div>
      <div class="result-section"><strong>Patient education</strong><ul id="edu-list"></ul></div>
      <div class="result-section" id="cautions"></div>
    </div>
  </section>
</main>
<footer>TBScreen · MobileNetV3-ONNX + local GGUF · decision support only, not diagnosis</footer>
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
  dropzone.ondragover=e=>{e.preventDefault(); dropzone.style.borderColor="var(--primary)"; dropzone.style.background="var(--primary-light)";};
  dropzone.ondragleave=e=>{dropzone.style.borderColor=""; dropzone.style.background="";};
  dropzone.ondrop=e=>{e.preventDefault(); dropzone.style.borderColor=""; dropzone.style.background=""; if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);};
  fileInput.onchange=e=>{ if(e.target.files[0]) handleFile(e.target.files[0]); };

  function handleFile(file){
    selectedFile=file;
    const reader=new FileReader();
    reader.onload=ev=>{ preview.src=ev.target.result; preview.style.display="block"; btnAnalyze.disabled=false; dropzone.innerHTML='<span class="dropzone-icon">✅</span>'+file.name; };
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
    $("triage").textContent="Triage: "+(data.triage||"—").toUpperCase()+" · Risk: "+(data.risk_level||"—");
    const interp=data.interpretation||{};
    $("body-text").innerHTML=withCitations(interp.interpretation||"");
    $("rec-text").innerHTML=withCitations(interp.recommendation||"");
    $("edu-list").innerHTML=(interp.education||[]).map(p=>"<li>"+withCitations(p)+"</li>").join("");
    $("cautions").innerHTML="<strong>Cautions</strong>"+(interp.cautions||[]).map(c=>'<div class="caution">'+escapeHtml(c)+"</div>").join("");
  }

  function renderQA(data){
    if(data.error){ alert(data.error); return; }
    empty.style.display="none"; results.style.display="block";
    $("prob").textContent="Clinical Q&A";
    $("triage").textContent="Grounded in offline WHO passages";
    const a=data.answer||{};
    $("body-text").innerHTML=withCitations(a.answer||"");
    $("rec-text").innerHTML=withCitations(a.recommendation||"");
    $("edu-list").innerHTML=(a.education||[]).map(p=>"<li>"+withCitations(p)+"</li>").join("");
    $("cautions").innerHTML="<strong>Cautions</strong>"+(a.cautions||[]).map(c=>'<div class="caution">'+escapeHtml(c)+"</div>").join("");
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
