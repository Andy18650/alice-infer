# tests/

- `test_*.cpp` — C++ (GoogleTest) engine tests, built as `alice_tests`.
  Slow end-to-end tests are gated on the presence of the bring-up model and
  fixtures (they `GTEST_SKIP` otherwise): `Model.RealBringUpSmoke`,
  `Canonical.GreedyMatchesHF`, `Tokenizer.BatteryMatchesHF`.
- `python/` — Python tooling (run with the project venv, `.venv`):
  - `dump_canonical.py` — regenerates the canonical PyTorch reference
    fixtures under `tests/canonical/` (tokenizer battery, greedy generations,
    fp32 logits). Run after intentionally changing tokenizer/model behavior
    or bumping the reference environment.
- `canonical/` — committed reference fixtures produced by the dumper.
  Logits are a bf16-compute reference: C++ tests assert exact token-id match
  on decisive steps (near-ties use a tie band; one step in case 1 is an
  exact tie in the reference itself) and within-tolerance logit drift.
