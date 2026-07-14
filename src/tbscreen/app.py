#!/usr/bin/env python3
"""Local Flask web application serving a premium, responsive, glassmorphic clinical dashboard."""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any

from flask import Flask, jsonify, request, render_template_string

# Ensure src/ is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tbscreen import TBScreenAssistant

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global assistant instance, lazy loaded
assistant: TBScreenAssistant | None = None


def get_assistant() -> TBScreenAssistant:
    """Lazy initialize the assistant to conserve resources until request."""
    global assistant
    if assistant is None:
        assistant = TBScreenAssistant()
    return assistant


# --- HTML Template ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TBScreen — Offline Clinical TB Assistant</title>
    <style>
        :root {
            --bg-color: #080B11;
            --card-bg: rgba(17, 22, 34, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #F3F4F6;
            --text-secondary: #9CA3AF;
            
            /* Curated HSL colors */
            --color-primary: hsl(210, 100%, 60%);     /* Electric Blue */
            --color-success: hsl(142, 70%, 45%);     /* Healing Green */
            --color-warning: hsl(38, 92%, 50%);      /* Amber Alert */
            --color-danger: hsl(4, 90%, 58%);        /* Crimson Red */
            
            --shadow-glow: 0 0 25px rgba(59, 130, 246, 0.15);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.04) 0%, transparent 40%);
        }

        header {
            padding: 1.5rem 2rem;
            border-bottom: 1px solid var(--border-color);
            backdrop-filter: blur(12px);
            background-color: rgba(8, 11, 17, 0.8);
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-weight: 700;
            font-size: 1.5rem;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #3B82F6 0%, #10B981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .badge-offline {
            background-color: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--color-success);
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .badge-offline::before {
            content: '';
            display: inline-block;
            width: 6px;
            height: 6px;
            background-color: var(--color-success);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--color-success);
        }

        main {
            flex: 1;
            padding: 2rem;
            max-width: 1400px;
            width: 100%;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr;
            gap: 2rem;
        }

        @media (min-width: 1024px) {
            main {
                grid-template-columns: 450px 1fr;
            }
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(16px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .upload-section {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 3rem 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.25s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--color-primary);
            background-color: rgba(59, 130, 246, 0.03);
            box-shadow: var(--shadow-glow);
        }

        .dropzone svg {
            width: 48px;
            height: 48px;
            color: var(--text-secondary);
            transition: color 0.25s ease;
        }

        .dropzone:hover svg {
            color: var(--color-primary);
        }

        .dropzone-text h3 {
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 0.25rem;
        }

        .dropzone-text p {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        #file-input {
            display: none;
        }

        .preview-container {
            display: none;
            width: 100%;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            position: relative;
            aspect-ratio: 1;
            background-color: #000;
        }

        .preview-image {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .btn-analyze {
            width: 100%;
            background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
            border: none;
            color: white;
            padding: 0.85rem;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .btn-analyze:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        }

        .btn-analyze:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        /* Results dashboard */
        .results-section {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            min-height: 400px;
            justify-content: center;
        }

        .results-empty {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            color: var(--text-secondary);
            gap: 1rem;
        }

        .results-empty svg {
            width: 64px;
            height: 64px;
            opacity: 0.4;
        }

        .results-content {
            display: none;
            flex-direction: column;
            gap: 1.5rem;
            animation: fadeIn 0.4s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .lang-bar {
            display: flex;
            gap: 0.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1rem;
        }

        .btn-lang {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 0.4rem 0.85rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-lang:hover {
            background-color: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
        }

        .btn-lang.active {
            background-color: var(--color-primary);
            border-color: var(--color-primary);
            color: white;
        }

        .dashboard-header {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }

        @media (min-width: 640px) {
            .dashboard-header {
                grid-template-columns: auto 1fr;
            }
        }

        /* Circular Progress Ring */
        .ring-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            position: relative;
            width: 160px;
            height: 160px;
        }

        .progress-ring {
            transform: rotate(-90deg);
        }

        .progress-ring__circle-bg {
            fill: transparent;
            stroke: rgba(255, 255, 255, 0.05);
            stroke-width: 12;
        }

        .progress-ring__circle {
            fill: transparent;
            stroke-width: 12;
            stroke-linecap: round;
            transition: stroke-dashoffset 0.8s ease-in-out, stroke 0.3s;
        }

        .ring-text {
            position: absolute;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .ring-percentage {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
        }

        .ring-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Triage status card */
        .triage-card {
            border-radius: 12px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .triage-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.75px;
            font-weight: 600;
        }

        .triage-value {
            font-size: 1.75rem;
            font-weight: 700;
            line-height: 1.1;
        }

        .triage-desc {
            font-size: 0.85rem;
        }

        /* Interpretation and Guidelines */
        .report-section {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }

        @media (min-width: 768px) {
            .report-section {
                grid-template-columns: 1fr 1fr;
            }
        }

        .section-title {
            font-size: 0.95rem;
            text-transform: uppercase;
            color: var(--text-secondary);
            letter-spacing: 0.75px;
            margin-bottom: 0.75rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .clinical-text {
            line-height: 1.6;
            font-size: 1rem;
        }

        .source-tag {
            display: inline-flex;
            align-items: center;
            background-color: rgba(59, 130, 246, 0.1);
            color: #60A5FA;
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 0.25rem;
            border: 1px solid rgba(59, 130, 246, 0.2);
            cursor: help;
        }

        .checklist {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .checklist-item {
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            line-height: 1.5;
            font-size: 0.95rem;
        }

        .checklist-item svg {
            width: 18px;
            height: 18px;
            margin-top: 2px;
            flex-shrink: 0;
        }

        /* Warnings and disclaimer */
        .cautions-box {
            background-color: rgba(239, 68, 68, 0.03);
            border: 1px solid rgba(239, 68, 68, 0.15);
            border-radius: 12px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .caution-item {
            font-size: 0.85rem;
            color: #F87171;
            line-height: 1.5;
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
        }

        .caution-item::before {
            content: '⚠️';
            font-size: 0.8rem;
        }

        /* Spinner / Loader */
        .loader {
            display: none;
            flex-direction: column;
            align-items: center;
            gap: 1.5rem;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(255, 255, 255, 0.05);
            border-top-color: var(--color-primary);
            border-radius: 50%;
            animation: spin 1s infinite linear;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        footer {
            padding: 1.5rem;
            text-align: center;
            font-size: 0.8rem;
            color: var(--text-secondary);
            border-top: 1px solid var(--border-color);
            margin-top: auto;
        }
    </style>
</head>
<body>

    <header>
        <div class="logo">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #3B82F6">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
            TBScreen AI
        </div>
        <div class="badge-offline">Offline Mode Enabled</div>
    </header>

    <main>
        <!-- Left Column: Upload -->
        <div class="card upload-section">
            <h2 style="font-size: 1.25rem; font-weight: 600;">Patient X-Ray Ingestion</h2>
            
            <div class="dropzone" id="dropzone">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>
                    <circle cx="9" cy="9" r="2"/>
                    <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
                </svg>
                <div class="dropzone-text">
                    <h3>Drag & Drop Chest X-Ray</h3>
                    <p>Supported: PNG, JPEG (Max 16MB)</p>
                </div>
            </div>
            
            <input type="file" id="file-input" accept="image/png, image/jpeg, image/jpg">
            
            <div class="preview-container" id="preview-container">
                <img src="" alt="X-ray Preview" class="preview-image" id="preview-image">
            </div>
            
            <button class="btn-analyze" id="btn-analyze" disabled>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="8"/>
                    <path d="m21 21-4.3-4.3"/>
                </svg>
                Screen & Interpret Patient
            </button>
        </div>

        <!-- Right Column: Results Dashboard -->
        <div class="card results-section">
            <!-- Empty state -->
            <div class="results-empty" id="results-empty">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
                    <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
                    <polyline points="14 2 14 8 20 8"/>
                    <line x1="8" y1="13" x2="16" y2="13"/>
                    <line x1="8" y1="17" x2="14" y2="17"/>
                </svg>
                <h3>Clinical Analysis Dashboard</h3>
                <p>Upload a patient chest X-ray and click analyze to generate an offline, RAG-grounded report.</p>
            </div>

            <!-- Loader state -->
            <div class="loader" id="loader">
                <div class="spinner"></div>
                <div style="text-align: center;">
                    <h3 style="font-weight: 500; margin-bottom: 0.25rem;">Running Offline AI Pipeline</h3>
                    <p style="font-size: 0.85rem; color: var(--text-secondary)">ONNX Screening + Local GGUF RAG Inference...</p>
                </div>
            </div>

            <!-- Content state -->
            <div class="results-content" id="results-content">
                <div class="lang-bar">
                    <button class="btn-lang active" data-lang="English">English</button>
                    <button class="btn-lang" data-lang="Yoruba">Yorùbá</button>
                    <button class="btn-lang" data-lang="Hausa">Hausa</button>
                    <button class="btn-lang" data-lang="Igbo">Igbo</button>
                </div>

                <div class="dashboard-header">
                    <!-- Gauge -->
                    <div class="ring-container">
                        <svg class="progress-ring" width="160" height="160">
                            <circle class="progress-ring__circle-bg" cx="80" cy="80" r="70"/>
                            <circle class="progress-ring__circle" id="gauge-bar" cx="80" cy="80" r="70"/>
                        </svg>
                        <div class="ring-text">
                            <span class="ring-percentage" id="prob-val">0%</span>
                            <span class="ring-label">Probability</span>
                        </div>
                    </div>

                    <!-- Triage card -->
                    <div class="triage-card" id="triage-box">
                        <span class="triage-title">Triage Classification</span>
                        <span class="triage-value" id="triage-val">-</span>
                        <p class="triage-desc" id="triage-desc"></p>
                    </div>
                </div>

                <div class="report-section">
                    <!-- Interpretation -->
                    <div>
                        <h4 class="section-title">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--color-primary)">
                                <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/>
                                <path d="M12 16v-4"/>
                                <path d="M12 8h.01"/>
                            </svg>
                            Clinical Interpretation
                        </h4>
                        <p class="clinical-text" id="interpret-text"></p>
                    </div>

                    <!-- Next steps -->
                    <div>
                        <h4 class="section-title">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--color-success)">
                                <path d="M9 11 12 14 22 4"/>
                                <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
                            </svg>
                            Recommended Next Steps
                        </h4>
                        <p class="clinical-text" id="recommend-text" style="font-weight: 500;"></p>
                    </div>
                </div>

                <!-- Patient Education -->
                <div>
                    <h4 class="section-title">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: var(--color-warning)">
                            <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
                            <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
                        </svg>
                        Patient Education points
                    </h4>
                    <ul class="checklist" id="education-list"></ul>
                </div>

                <!-- Cautions box -->
                <div class="cautions-box" id="cautions-box"></div>
            </div>
        </div>
    </main>

    <footer>
        TBScreen AI Assistant • Powered by MobileNetV3-ONNX & Gemma-4-E2B-GGUF • 100% Offline Decision Support
    </footer>

    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('file-input');
        const previewContainer = document.getElementById('preview-container');
        const previewImage = document.getElementById('preview-image');
        const btnAnalyze = document.getElementById('btn-analyze');
        
        const resultsEmpty = document.getElementById('results-empty');
        const loader = document.getElementById('loader');
        const resultsContent = document.getElementById('results-content');
        
        let selectedFile = null;
        let currentLang = "English";

        // Drag & Drop
        dropzone.addEventListener('click', () => fileInput.click());
        
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });
        
        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });
        
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                dropzone.style.display = 'none';
                previewContainer.style.display = 'block';
                btnAnalyze.disabled = false;
            };
            reader.readAsDataURL(file);
        }

        // Analysis
        btnAnalyze.addEventListener('click', () => {
            if (!selectedFile) return;
            
            resultsEmpty.style.display = 'none';
            resultsContent.style.display = 'none';
            loader.style.display = 'flex';
            btnAnalyze.disabled = true;

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('lang', currentLang);

            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                loader.style.display = 'none';
                btnAnalyze.disabled = false;
                if (data.error) {
                    alert('Error: ' + data.error);
                    resultsEmpty.style.display = 'flex';
                    return;
                }
                renderResults(data);
            })
            .catch(err => {
                loader.style.display = 'none';
                btnAnalyze.disabled = false;
                alert('Connection error or model load failure.');
                resultsEmpty.style.display = 'flex';
            });
        });

        // Language toggle
        document.querySelectorAll('.btn-lang').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.btn-lang').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentLang = btn.getAttribute('data-lang');
                
                // If results are currently showing, translate them on the fly
                if (resultsContent.style.display === 'flex' || resultsContent.style.display === 'block') {
                    resultsContent.style.opacity = '0.5';
                    fetch('/translate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ lang: currentLang })
                    })
                    .then(res => res.json())
                    .then(data => {
                        resultsContent.style.opacity = '1';
                        if (!data.error) {
                            renderResults(data);
                        }
                    })
                    .catch(() => {
                        resultsContent.style.opacity = '1';
                    });
                }
            });
        });

        function renderResults(data) {
            resultsEmpty.style.display = 'none';
            loader.style.display = 'none';
            resultsContent.style.display = 'flex';

            const prob = data.vision_result.tb_probability;
            const probPercent = Math.round(prob * 100);
            
            // Render percentage
            document.getElementById('prob-val').innerText = probPercent + '%';

            // Circular progress bar
            const circle = document.getElementById('gauge-bar');
            const radius = circle.r.baseVal.value;
            const circumference = radius * 2 * Math.PI;
            circle.style.strokeDasharray = `${circumference} ${circumference}`;
            
            // Calibrate gauge color and offset
            let color = 'var(--color-success)';
            if (data.risk_level === 'high') {
                color = 'var(--color-danger)';
            } else if (data.risk_level === 'moderate') {
                color = 'var(--color-warning)';
            }
            
            circle.style.stroke = color;
            const offset = circumference - (prob * circumference);
            circle.style.strokeDashoffset = offset;

            // Render Triage Card
            const triageBox = document.getElementById('triage-box');
            const triageVal = document.getElementById('triage-val');
            const triageDesc = document.getElementById('triage-desc');
            
            triageVal.innerText = data.triage.toUpperCase();
            
            if (data.triage === 'refer') {
                triageBox.style.backgroundColor = 'rgba(239, 68, 68, 0.1)';
                triageBox.style.border = '1px solid rgba(239, 68, 68, 0.2)';
                triageVal.style.color = 'var(--color-danger)';
                triageDesc.innerText = 'High risk detected. Immediate referral for confirmatory bacteriological testing (Xpert MTB/RIF) required.';
            } else if (data.triage === 'retest') {
                triageBox.style.backgroundColor = 'rgba(245, 158, 11, 0.1)';
                triageBox.style.border = '1px solid rgba(245, 158, 11, 0.2)';
                triageVal.style.color = 'var(--color-warning)';
                triageDesc.innerText = 'Borderline screening result. Retest patient or follow up closely based on clinical presentation.';
            } else {
                triageBox.style.backgroundColor = 'rgba(16, 185, 129, 0.1)';
                triageBox.style.border = '1px solid rgba(16, 185, 129, 0.2)';
                triageVal.style.color = 'var(--color-success)';
                triageDesc.innerText = 'Negative screening result. TB disease is unlikely. Monitor if symptoms persist.';
            }

            // Render Interpretation & Recommendation
            const interpret = data.interpretation || {};
            document.getElementById('interpret-text').innerHTML = formatCitations(interpret.interpretation || 'No explanation generated.');
            document.getElementById('recommend-text').innerHTML = formatCitations(interpret.recommendation || 'No recommendation.');

            // Render Education
            const eduList = document.getElementById('education-list');
            eduList.innerHTML = '';
            (interpret.education || []).forEach(point => {
                const li = document.createElement('li');
                li.className = 'checklist-item';
                li.innerHTML = `
                    <svg viewBox="0 0 24 24" fill="none" stroke="var(--color-success)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="20 6 9 17 4 12"/>
                    </svg>
                    <span>${formatCitations(point)}</span>
                `;
                eduList.appendChild(li);
            });

            // Render Cautions Box
            const cautionsBox = document.getElementById('cautions-box');
            cautionsBox.innerHTML = '<span style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #F87171; letter-spacing: 0.5px;">Safety Warnings & Disclaimers</span>';
            (interpret.cautions || []).forEach(caution => {
                const div = document.createElement('div');
                div.className = 'caution-item';
                div.innerText = caution;
                cautionsBox.appendChild(div);
            });
        }

        function formatCitations(text) {
            // Replaces e.g. [who-tb-screening-01] with a styled source tag
            return text.replace(/\\[([a-zA-Z0-9\\-]+)\\]/g, (match, id) => {
                return `<span class="source-tag" title="Grounding Source">${id}</span>`;
            });
        }
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
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1]
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    lang = request.form.get("lang", "English")

    try:
        assist = get_assistant()
        result = assist.process_image(filepath, lang=lang)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/translate", methods=["POST"])
def translate():
    """Re-interpret cached vision result in a new language — skips ONNX."""
    data = request.get_json() or {}
    lang = data.get("lang", "English")

    try:
        assist = get_assistant()
        result = assist.reinterpret(lang=lang)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """Grounded clinical Q&A (matches metadata.json test-prompt style)."""
    data = request.get_json() or {}
    question = (data.get("question") or "").strip()
    lang = data.get("lang", "English")
    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        assist = get_assistant()
        result = assist.ask(question, lang=lang)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    host = os.environ.get("TBSCREEN_HOST", "127.0.0.1")
    port = int(os.environ.get("TBSCREEN_PORT", "5000"))
    print(f"Starting TBScreen Clinical Dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
