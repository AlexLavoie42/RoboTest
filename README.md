# RoboTest – Simulated Conscious Agent 🚀

A research playground for building an **embodied AI agent** in **NVIDIA Omniverse Isaac Sim** that implements cognitive modules inspired by theories of consciousness (GNWT, HOT, AST).

![diagram](docs/img/architecture_overview.png)

## Quick Start
```bash
# 1. Clone
$ git clone https://github.com/AlexLavoie42/RoboTest.git && cd RoboTest

# 2. Launch stack (needs local GPU & Isaac Sim assets)
$ docker compose up -d

# 3. Run smoke test (once Phase 6 tasks complete)
$ python tests/e2e_smoke.py

Status: Early scaffolding. Follow the project board for current sprint.

---
## CONTRIBUTING.md
```markdown
# Contributing Guide

Thank you for helping! We follow a **fork→branch→PR** workflow.

1. **Fork** the repo & create a feature branch (`feat/your-feature`).
2. Ensure `pre-commit run --all-files` passes.
3. Add / update **unit tests** (PyTest) for any new code.
4. Submit a Pull Request. The PR template will ask for:
   - Linked issue
   - Description & screenshots
   - Checklist – lint/format, tests, docs
5. At least one reviewer (ChatGPT 🤖 or Alex) will approve.

### Coding Standards
- **Python 3.10**
- Black + isort formatting
- Type‑hints required for new modules
- Docstrings must follow Google style

### Commit Messages
Use Conventional Commits (`feat:`, `fix:`, `docs:` …).

### Branch Protection
`main` is protected. CI must pass before merge.