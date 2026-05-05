# Archive Plan

This repository cleanup removes `archive/` from `main`.

Recommended next step:

- preserve the pre-cleanup state on a dedicated branch such as `full-experiment-history`
- keep `main` focused on the active package, saved headline results, and resume-facing documentation

Safety note:

- confirm the working `main` commit has already been pushed
- create the history-preserving branch before removing `archive/`

Status for this cleanup pass:

- local `main` was confirmed to match `origin/main` before deletion
- a local branch named `full-experiment-history` was created as a safety branch
- `archive/` was then removed from `main`
