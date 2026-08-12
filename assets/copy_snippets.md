# Webpage copy snippets for UserToolBench

## One-sentence description
UserToolBench is a profile-hidden benchmark for evaluating whether tool-use LLMs can infer user preferences from interaction history and make correct personalized tool-call decisions under incomplete information.

## Short abstract for website
Tool-use LLMs are increasingly expected to act on behalf of persistent users. UserToolBench evaluates this setting by hiding the explicit user profile during evaluation: the model sees only the accumulated interaction history, the current request, and available tool schemas. Success requires choosing the right tools, arguments, clarification behavior, and multi-step trajectory for the particular user, not merely completing the generic task.

## Key claims
- Personalization should be evaluated through decisions, not only response style.
- Exact tool-call trajectory matching provides a high-precision proxy for profile-conditioned decision alignment.
- Relaxed task completion can overestimate personalization: many predictions are executable but not user-aligned.
- Multi-tool coordination, missing-constraint inference, and long-horizon consistency remain major bottlenecks.

## Suggested GitHub Pages structure
```
index.html
assets/
  profile_hidden_pipeline.svg
  dataset_composition.svg
  performance_gap.svg
  bottlenecks.svg
paper.pdf
```
