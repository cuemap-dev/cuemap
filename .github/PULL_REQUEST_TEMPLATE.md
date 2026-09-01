## Summary

<!-- What problem does this change solve? -->

## Changes

<!-- List the important implementation or documentation changes. -->

## Testing

- [ ] `cargo check --locked --all-targets`
- [ ] `cargo test --locked`
- [ ] Relevant Windows, package, or integration checks were run or are covered by CI.
- [ ] For code changes, the lexical and hybrid release benchmarks were rerun and the hybrid recall average remains below 10 ms at every requested scale; results are included below.
- [ ] Candidate generation remains semantic-model-free; semantic models are used only for bounded post-generation reranking, with regression coverage if that boundary changed.

## Release impact

- [ ] No user-facing behavior change
- [ ] README/changelog updated
- [ ] Migration or compatibility notes included
- [ ] Version/package/release workflow impact reviewed

## Notes

<!-- Mention known limitations, follow-ups, or platform-specific details. -->

## Benchmark results

<!-- Required for code changes covered by CONTRIBUTING.md. Include hardware, commands, and lexical/hybrid average, P50, and P95 results. -->
