# Deploy QuantMind to Hugging Face Spaces

## Two-Space workflow (prod + staging)

We run **two** HF Spaces to keep production stable while features are being tested:

| Space | URL | Tracks | Purpose |
|---|---|---|---|
| Production | https://huggingface.co/spaces/qicaixin/ai-helper | `master` branch | User-facing; only updated after PR merge |
| Staging | https://huggingface.co/spaces/qicaixin/ai-helper-staging | feature branch under test | QA before merge |

### Golden rules
1. **Never commit directly to `master`.** All changes go via PRs from `feat/*` or `fix/*` branches.
2. **Deploy the feature branch to staging first.** Validate end-to-end on the staging Space.
3. **Only after the user confirms it works**, merge the PR and deploy `master → prod`.

## Prerequisites

- [Git](https://git-scm.com/) installed
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) logged in (`huggingface-cli login`)
- Push access to both HF Space repos

## One-Time Setup

### 1. Add both HF Spaces as git remotes

```bash
cd C:\qlik\tools\kronos-qlib-toolkit
git remote add hf         https://huggingface.co/spaces/qicaixin/ai-helper
git remote add hf-staging https://huggingface.co/spaces/qicaixin/ai-helper-staging
```

Verify remotes:

```bash
git remote -v
# qicaixin    https://github.com/qicaixin/quantmind.git (fetch/push)
# hf          https://huggingface.co/spaces/qicaixin/ai-helper (fetch/push)
# hf-staging  https://huggingface.co/spaces/qicaixin/ai-helper-staging (fetch/push)
```

### 2. Copy secrets to the staging Space

The `duplicate_space` SDK call copies the Dockerfile and hardware tier but **not the secrets**.
Open https://huggingface.co/spaces/qicaixin/ai-helper-staging/settings and add the same
environment variables / secrets that the prod Space has (LLM API keys, etc).

### 3. Key files required by HF Spaces

| File | Purpose |
|---|---|
| `Dockerfile` | Tells HF how to build and run the app (python:3.12-slim, port 7080) |
| `README.md` | YAML frontmatter at the top configures the Space (`sdk: docker`, `app_port: 7080`) |
| `app.py` | Flask must bind to `0.0.0.0` (not `127.0.0.1`) for Docker container access |

**README.md frontmatter:**

```yaml
---
title: QuantMind AI Helper
emoji: 🔮
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7080
---
```

## Deploy a feature for testing (staging)

While iterating on a `feat/*` branch:

```bash
# Push the feature branch to GitHub as usual
git push qicaixin feat/my-feature

# Deploy the SAME branch to the staging Space (mapped to its main branch)
git push hf-staging feat/my-feature:main --force
```

Monitor build at https://huggingface.co/spaces/qicaixin/ai-helper-staging.
Iterate on the branch — every push to `hf-staging` redeploys staging only.
Production stays untouched.

## Deploy to production (only after PR merge)

```bash
# 1. Ensure master is up to date with the merged PR
git checkout master
git pull qicaixin master

# 2. Push master to prod (mapped to main)
git push hf master:main --force
```

Open https://huggingface.co/spaces/qicaixin/ai-helper and watch the **Build** logs.
The Docker build typically takes 3–5 minutes.

## Troubleshooting

| Issue | Solution |
|---|---|
| Build fails on `pip install` | Check `requirements.txt` for version conflicts. HF Spaces uses Linux — Windows-only packages won't work. |
| App shows "Building" forever | Click the **Logs** tab on the Space page to see build errors. |
| App starts but shows blank/error | Check **Container Logs** on HF. Common issue: `host="127.0.0.1"` instead of `host="0.0.0.0"` in `app.py`. |
| `git push hf` asks for credentials | Run `huggingface-cli login` and enter your HF token. |
| Port mismatch | Ensure `app_port` in README.md YAML matches the port in `app.py` (currently both 7080). |
| Staging works but prod doesn't | Likely a missing secret — compare env vars between the two Spaces' Settings pages. |

## Architecture Notes

- **GitHub `qicaixin/quantmind`** is the source of truth for code (PR-based workflow)
- **`hf-staging`** receives feature-branch force-pushes for QA
- **`hf` (prod)** only receives `master:main` after a PR has been merged
- The Dockerfile copies the entire repo into `/app` and runs `python app.py`
- SQLite DB and outputs are ephemeral in the container (reset on each redeploy)
- Secrets (API keys) are configured per-Space via the Settings UI; staging needs its own set

