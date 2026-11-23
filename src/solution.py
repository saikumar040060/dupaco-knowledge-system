"""
Entry point: `build()` returns an object with ingest/answer/conflicts that
implements the KnowledgeSystem protocol in contract.py. See DESIGN.md for the
design write-up; in short:

  ingest()    -> parse every page's tables and entity facts into a versioned
                 store (ingest.py / tables.py / facts.py). Never overwrites.
  answer()    -> route the question to a structured lookup first, a
                 retrieval fallback second (qa.py), resolved against a
                 single written-down "what's current" policy (policy.py).
  conflicts() -> re-scan the same store for genuine disagreements
                 (conflicts.py).

No network calls, no LLM, no randomness — every answer is reproducible from
the snapshots alone, so paraphrases are consistent and answering stays fast.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import Snapshot  # noqa: E402

import conflicts  # noqa: E402
from ingest import Store  # noqa: E402
from qa import answer_question  # noqa: E402


class DupacoKnowledge:
    def __init__(self):
        self.store = Store()

    def ingest(self, snapshot: Snapshot) -> None:
        self.store.ingest(snapshot)

    def answer(self, question: str, as_of=None):
        return answer_question(self.store, question, as_of)

    def conflicts(self):
        return conflicts.all_conflicts(self.store)


def build():
    return DupacoKnowledge()
