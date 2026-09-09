# tests/

- `test_*.cpp` — C++ (GoogleTest) engine tests, built as `alice_tests`.
  Slow end-to-end tests are gated on the presence of the bring-up model and
  fixtures (they `GTEST_SKIP` otherwise): `Model.RealBringUpSmoke`,
  `Canonical.TeacherForcedLogitsMatchHF`, `Tokenizer.BatteryMatchesHF`.
- `python/` — Python tooling (run with the project venv, `.venv`):
  - `dump_canonical.py` — regenerates the canonical PyTorch reference
    fixtures under `tests/canonical/` (tokenizer battery, greedy generations,
    fp32 logits). Run after intentionally changing tokenizer/model behavior
    or bumping the reference environment.
- `canonical/` — committed reference fixtures produced by the dumper.
  Generation cases are TEACHER-FORCED: the reference sequence (prompt +
  greedy continuation) is replayed in a single forward pass and logits at
  every position are recorded (`case_<i>.teacher.f32`, float32
  `[seq_len, vocab]`). The C++ test replays the same sequence and compares
  logit vectors only — no argmax assertions, so reference near-ties are
  irrelevant by construction. Logits are a bf16-compute reference; the
  assertion is max drift < 0.75 (observed ~0.2) with printed metrics.
