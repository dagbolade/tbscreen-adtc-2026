#!/usr/bin/env python3
"""Unit tests for Unicode tokenization, topic retrieval, and WHO triage policy."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tbscreen import core
from tbscreen.rag import Retriever, tokenize


class TestUnicodeTokenize(unittest.TestCase):
    def test_yoruba_diacritics(self):
        toks = tokenize("Àyẹ̀wò ikọ́ fée fée ọ̀sẹ̀")
        joined = " ".join(toks)
        self.assertIn("àyẹ̀wò", toks)
        self.assertIn("ọ̀sẹ̀", toks)
        self.assertIn("ikọ́", toks)
        self.assertTrue("ọ" in joined and "ẹ" in joined)

    def test_igbo_characters(self):
        toks = tokenize("ọrịa ụkwara nke igbu mmadụ")
        self.assertIn("ọrịa", toks)
        self.assertIn("ụkwara", toks)


class TestTopicRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        corpus = os.path.join(ROOT, "corpus", "sources", "who_tb_guidelines.jsonl")
        cls.retriever = Retriever.from_passages(
            [json.loads(l) for l in open(corpus, encoding="utf-8") if l.strip()]
        )

    def test_yoruba_refer_topics(self):
        hits = self.retriever.retrieve_by_topics(
            ["triage_refer", "screening"], lang="Yoruba", k=3, query="triage_refer screening"
        )
        self.assertTrue(hits)
        self.assertTrue(all(h["lang"] == "yo" for h in hits))
        self.assertTrue(any(h.get("topic") == "triage_refer" for h in hits))

    def test_hausa_patient_education(self):
        hits = self.retriever.retrieve_by_topics(
            ["patient_education"], lang="Hausa", k=2, query="patient_education"
        )
        self.assertTrue(all(h["lang"] == "ha" for h in hits))

    def test_igbo_clear_topics(self):
        hits = self.retriever.retrieve_by_topics(
            ["triage_clear", "screening"], lang="Igbo", k=3, query="triage_clear"
        )
        self.assertTrue(all(h["lang"] == "ig" for h in hits))


class TestTriagePolicy(unittest.TestCase):
    def test_negative_screen_does_not_clear_symptomatic(self):
        d = core.decide_triage(0.15, age_years=34, cough_weeks=3, has_tb_symptoms=True)
        self.assertEqual(d["triage"], "symptom_followup")
        self.assertEqual(d["screen_status"], "NEGATIVE")

    def test_high_prob_refers(self):
        d = core.decide_triage(0.82, age_years=40)
        self.assertEqual(d["triage"], "refer")

    def test_pediatric_out_of_scope(self):
        d = core.decide_triage(0.20, age_years=10)
        self.assertEqual(d["screen_status"], "OUT_OF_SCOPE")
        self.assertEqual(d["triage"], "refer")
        self.assertFalse(d["cad_in_scope"])

    def test_asymptomatic_low_monitors(self):
        d = core.decide_triage(0.12, age_years=30, has_tb_symptoms=False)
        self.assertEqual(d["triage"], "monitor")


class TestIndexRoundtrip(unittest.TestCase):
    def test_save_load(self):
        corpus = os.path.join(ROOT, "corpus", "sources", "who_tb_guidelines.jsonl")
        passages = [json.loads(l) for l in open(corpus, encoding="utf-8") if l.strip()]
        r = Retriever.from_passages(passages[:8])
        with tempfile.TemporaryDirectory() as tmp:
            r.save_index(tmp)
            r2 = Retriever.load_index(tmp, passages[:8])
            self.assertEqual(len(r2.passages), 8)


if __name__ == "__main__":
    unittest.main()
