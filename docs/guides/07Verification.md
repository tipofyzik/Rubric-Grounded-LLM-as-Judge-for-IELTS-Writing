# Verification and release


`python Manage.py test --suite portable` runs CPU contract checks without source datasets, pretrained weights, `.env` or hosted calls. `--suite all` also covers main, legacy notebook and deployment suites and requires their configured local fixtures. SQL contracts use the local JavaScript/PGlite harness. These checks do not measure grading quality.

For the final demo also verify: three examples, one fresh own input, input rejection, EN/RU/DE state preservation, stale-score clearing, session separation, queue behavior, restart and local fallback. Use controlled fixtures for provider faults rather than deliberately corrupting the production database. Record the exact source revision and which checks actually ran.

Before publishing, review the changes and check that the release contains the source, tests, guides and settings template. Keep credentials, model weights, private data and generated results outside the source release.

Run `python Manage.py source-zip output/ielts-demo-source.zip` into a new filename and inspect a clean extraction. The owner publishes the reviewed changes. The GitHub workflow becomes remote evidence only after that commit is pushed and its run succeeds; local tests do not establish remote CI success for an unpublished commit.

[Documentation catalogue](../README.md)
