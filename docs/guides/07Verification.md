# Verification and release


`python Manage.py test --suite portable` runs CPU contract checks without source datasets, pretrained weights, `.env` or hosted calls. `--suite all` also covers main, legacy notebook and deployment suites and requires their configured local fixtures. SQL contracts use the local JavaScript/PGlite harness. These checks do not measure grading quality.

For the final demo also verify: three examples, one fresh own input, input rejection, EN/RU/DE state preservation, stale-score clearing, session separation, queue behavior, restart and local fallback. Use controlled fixtures for provider faults rather than deliberately corrupting the production database. Record the exact source revision and which checks actually ran.

Before GitHub publication, review the deny-all allowlist and `git status`; previously tracked paths remain tracked despite ignore changes. Include package `__init__.py`, tests, safe templates, current guides and final evaluation/acceptance summaries. Intermediate stage reports, old runbooks, inventory audits and refactoring notes remain local. Exclude `.env`, weights, data, generated results, caches, archived originals and the local `.gitignore`.

Run `python Manage.py source-zip output/ielts-demo-source.zip` into a new filename and inspect a clean extraction. The owner publishes the reviewed changes. The GitHub workflow becomes remote evidence only after that commit is pushed and its run succeeds; local tests do not establish remote CI success for an unpublished commit.

The current engineering release may finish while repeatability, robustness and new model experiments remain deferred. Do not mark the complete historical research roadmap finished on that basis. Existing baseline metrics and artifact provenance remain unchanged.


[Documentation catalogue](../README.md)
