# Vercel Deployment Guide

This project uses a static export flow for Vercel hosting.

## 1. Build static deploy folder

```bash
source .venv/bin/activate
python scripts/export_static_site.py --out-dir site
```

What this generates:
- `site/index.html` (full multi-page website shell)
- `site/data/payload.json` (dashboard data from latest lock or reports)
- `site/vercel.json` (rewrites for routes and API alias)

## 2. Deploy to Vercel

From `site/`:

```bash
cd site
vercel login
vercel --prod
```

If your Vercel project is already linked, `vercel --prod` is enough.

## 2b. Enable GitHub auto-deploy (recommended)

This repo includes:
- `.github/workflows/vercel-deploy.yml`

On push to `main`, GitHub Actions will:
1. export static site to `site/`
2. inject Vercel project metadata
3. deploy to Vercel production

Add these GitHub repository secrets:
- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

How to get them:
- `VERCEL_TOKEN`: Vercel dashboard -> Settings -> Tokens
- `VERCEL_ORG_ID` + `VERCEL_PROJECT_ID`:
  - run `vercel link` once locally in repo root
  - read IDs from `.vercel/project.json`

## 3. Route behavior

These routes are supported and rewritten to `index.html`:
- `/overview`
- `/races`
- `/simulator`
- `/integrity`

Data endpoint:
- `/api/data` -> `/data/payload.json`

## 4. Refresh data after new pipeline run

After generating new outputs or lock snapshots, re-run export and redeploy:

```bash
python scripts/export_static_site.py --out-dir site
cd site
vercel --prod
```

If using GitHub auto-deploy, just commit and push changes to `main`.

## 5. Optional: deploy specific snapshot

```bash
python scripts/export_static_site.py \
  --snapshot-dir reports/locks/<snapshot_id> \
  --out-dir site
```
