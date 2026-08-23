"""Automated tests"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tbscreen import core, pipeline  # noqa: E402
from tbscreen.rag import Retriever  # noqa: E402


class TestOnnxInference(unittest.TestCase):
    def test_predict_on_tracked_sample(self):
        sample = ROOT / "samples" / "normal_01.png"
        if not sample.exists():
            self.skipTest("samples/normal_01.png missing")
        from vision.inference import TBScreenModel

        model = TBScreenModel()
        out = model.predict(str(sample), with_zones=False)
        self.assertIn("tb_probability", out)
        self.assertGreaterEqual(out["tb_probability"], 0.0)
        self.assertLessEqual(out["tb_probability"], 1.0)
        self.assertIn(out["screening_result"], {"POSITIVE", "NEGATIVE"})


class TestPromptContracts(unittest.TestCase):
    def test_interpret_schema_keys(self):
        prompt = core.interpret_user_content("summary", "[who-tb-screening-01] text", "English")
        self.assertIn("interpretation", prompt)
        self.assertIn("sources", prompt)
        self.assertIn("not a diagnosis", prompt.lower() + " screening")

    def test_qa_schema_keys(self):
        prompt = core.qa_user_content("What is screening?", "[who-tb-screening-01] text", "Hausa")
        self.assertIn("answer", prompt)
        self.assertIn("Hausa", prompt)

    def test_qa_prompt_includes_screening_summary(self):
        prompt = core.qa_user_content(
            "Explain my result",
            "[who-tb-screening-01] text",
            "English",
            screening_summary="TB probability: 78.3%\nRisk level: high",
        )
        self.assertIn("TB probability: 78.3%", prompt)
        self.assertIn("patient screening context", prompt.lower())
        self.assertIn("their own case", prompt.lower())

    def test_qa_prompt_without_screening_summary(self):
        prompt = core.qa_user_content("What is screening?", "[who-tb-screening-01] text", "English")
        self.assertIn("no screening", prompt.lower())


class TestMetadataPromptPaths(unittest.TestCase):
    def test_metadata_has_two_prompts(self):
        meta = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(len(meta["test_prompts"]), 2)
        self.assertEqual(meta["model"]["runtime"], "llama.cpp")
        self.assertTrue(meta["_runtime"]["model_path"].endswith(".gguf"))

    def test_qa_path_uses_retriever(self):
        corpus = ROOT / "corpus" / "sources" / "who_tb_guidelines.jsonl"
        retriever = Retriever.from_jsonl(corpus)
        meta = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
        question = meta["test_prompts"][1]["prompt"]

        class FakeLLM:
            pass

        with mock.patch("tbscreen.pipeline.llm.answer_question") as mocked:
            mocked.return_value = {
                "answer": "Screening is not diagnosis.",
                "recommendation": "Refer for Xpert if screen-positive.",
                "education": ["TB is curable."],
                "cautions": ["This is decision support, not a diagnosis."],
                "sources": ["who-tb-screening-01"],
            }
            out = pipeline.answer_clinical_question(FakeLLM(), retriever, question, lang="English", k=3)
            self.assertTrue(out["retrieved_sources"])
            self.assertIn("answer", out["answer"])
            self.assertFalse(out["has_screen_context"])

    def test_qa_path_threads_screening_context(self):
        corpus = ROOT / "corpus" / "sources" / "who_tb_guidelines.jsonl"
        retriever = Retriever.from_jsonl(corpus)

        class FakeLLM:
            pass

        vision_result = {"tb_probability": 0.783, "screening_result": "POSITIVE"}
        with mock.patch("tbscreen.pipeline.llm.answer_question") as mocked:
            mocked.return_value = {
                "answer": "Personalized.",
                "recommendation": "Refer.",
                "education": ["TB is curable."],
                "cautions": ["This is decision support, not a diagnosis."],
                "sources": ["who-tb-screening-01"],
            }
            out = pipeline.answer_clinical_question(
                FakeLLM(),
                retriever,
                "Can you explain the results better to me?",
                vision_result=vision_result,
                patient_context={"age_years": 34},
            )
            self.assertTrue(out["has_screen_context"])
            summary = mocked.call_args.kwargs.get("screening_summary")
            self.assertIn("78.3%", summary)
            self.assertIn("refer", summary)
            # Triage-topic passages for this decision are merged into grounding.
            self.assertGreaterEqual(len(out["retrieved_sources"]), 1)


class TestOfflineIndex(unittest.TestCase):
    def test_index_artifacts_exist(self):
        self.assertTrue((ROOT / "corpus" / "index" / "tfidf.npz").exists())
        self.assertTrue((ROOT / "corpus" / "index" / "vocab.json").exists())


if __name__ == "__main__":
    unittest.main()
