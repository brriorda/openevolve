# Typed candidate rejection

This example evolves a small `square(value)` function. The evaluator returns
`CandidateRejected` when a proposal has invalid Python syntax, omits the required
function, or fails while running that function. A valid function returns a numeric
`combined_score` instead. It needs only OpenEvolve and an LLM provider key for a
full evolution run.

## See a rejection without an LLM call

From the OpenEvolve repository root, with the package installed (`pip install -e .`):

```bash
python -c 'from examples.typed_rejection.evaluator import evaluate; print(evaluate("examples/typed_rejection/rejected_program.py"))'
```

The fixture omits `square`, so the result has category `static_invalid` and code
`square_function_missing`. To compare a valid candidate:

```bash
python -c 'from examples.typed_rejection.evaluator import evaluate; print(evaluate("examples/typed_rejection/initial_program.py"))'
```

The valid seed returns a score rather than a rejection, even though its estimate
is inaccurate. A zero score alone does not mean candidate rejection.

## Run the typed evaluator rejection regression

```bash
pytest -q tests/test_typed_evaluator_rejection_attempt_ledger.py
```

The test scripts one invalid-interface proposal, evaluates it with this
example's evaluator, and checks the run-directory attempt ledger and baseline
rejection artifact. It makes no provider calls.

For categorical exclusion, run the separate one-iteration regression:

```bash
pytest -q tests/test_discard_only_smoke.py
```

It checks the durable attempt, the absence of a rejected child from selection
and checkpoint state, and admission of an accepted zero-score control.

## Run evolution

Set `OPENAI_API_KEY` for the configured OpenAI-compatible model, then run:

```bash
python openevolve-run.py \
  examples/typed_rejection/initial_program.py \
  examples/typed_rejection/evaluator.py \
  --config examples/typed_rejection/config.yaml \
  --iterations 8 \
  --output typed_rejection_output
```

Change `llm.models[0].name` in `config.yaml` and set `llm.api_base` if your
provider needs another model or endpoint. No particular run is guaranteed to
produce a rejected proposal. When one occurs, the controller writes a bounded
record to `typed_rejection_output/attempts/rejected_attempts.jsonl` with the
parent ID, category, code, rationale, proposal usage when available, and digests
of prompt and response text. Raw prompts, responses, and candidate code do not
enter that ledger.

With the `artifact_low_score` policy, an ordinary rejected proposal still follows the
existing low-score admission path and receives a `rejection` artifact. A typed
`integrity_rejected` outcome is a hard exclusion under every policy: its attempt is
recorded, but its candidate cannot become a selectable program, including after
adjudication. To run
the same example with categorical exclusion, use
`--config examples/typed_rejection/discard_only.yaml` and a distinct output
directory. Only `rejection_memory.policy` differs between the two config files.
Under `discard_only`, rejected proposals appear in the attempt ledger but do
not become programs or program-owned artifacts. An accepted score of zero is
still eligible for admission.

## Compare deferred feedback policies

Use `--config examples/typed_rejection/parent_next_once.yaml` to deliver an
eligible rejection diagnosis once to a later proposal selected from that same
admitted parent. The run-directory store reserves an attempt before worker
dispatch and writes its claim transition to
`attempts/feedback_claims.jsonl`. The completion record names the outcome and
prompt digest; it does not copy the full prompt or candidate code. The
`generation_format_invalid`, `static_invalid`, and `runtime_candidate_failure`
categories are eligible. Integrity, novelty, and operational failures are not.

Use `--config examples/typed_rejection/global_history.yaml` for the bounded
global-history comparator. It shows recent eligible diagnoses with their source
parent IDs, even if the selected parent differs. Both policies exclude rejected
candidates from the program population. The model may or may not produce a
rejection in eight iterations; use the attempt ledger to inspect what occurred.

The four example configs share every resolved setting except
`rejection_memory.policy`; this one field selects the baseline, discard-only,
parent-next-once, or global-history behavior. Run the independent offline smoke
test without TeamBench or a model provider:

```bash
pytest -q tests/test_parent_feedback_smoke.py
```

It checks one-shot delivery, parent isolation, global context, claim persistence,
hard integrity exclusion, and the paired config invariant with scripted proposals.
