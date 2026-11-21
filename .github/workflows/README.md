# CI runner recovery

`tests.yml` keeps the Linux and macOS ARM64 test matrix. A failed test, dependency
installation, or wheel check must remain visible.

`retry-runner-startup.yml` requests one automatic retry when GitHub confirms that
a cancelled macOS job never acquired a hosted runner. It checks runner metadata,
executed steps, provisioning annotations, the current commit, and the run attempt.
It leaves successful jobs alone and does not retry real test failures, ordinary
user cancellations, or superseded runs.

The recovery workflow runs on Ubuntu and reads only GitHub metadata and trusted
default-branch code. It never checks out the triggering PR or consumes its artifacts.

GitHub only activates `workflow_run` workflows when they exist on the default
branch, so this protection starts after the CI change is merged into `master`.
Recovery happens after a failed run; it cannot prevent GitHub's initial failure
notification or guarantee runner capacity during an outage. A failed retry stays
red and requires investigation.

Run the recovery policy tests without installing TensorFlow:

```sh
node --test .github/scripts/retry-runner-startup.test.cjs
```
