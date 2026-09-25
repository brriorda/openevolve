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

With the `artifact_low_score` policy, a rejected proposal still follows the
existing low-score admission path and receives a `rejection` artifact. This
example does not demonstrate a policy that excludes rejected proposals.
