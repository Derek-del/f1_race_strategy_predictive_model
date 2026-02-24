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

## 2b. Connect GitHub for auto deploy

In Vercel:
1. Add New Project
2. Import your GitHub repo
3. Set `Root Directory` to `site`
4. Keep framework as `Other`
5. Build command can be empty (static site already exported)
6. Deploy

After that, every push to `main` triggers a new Vercel deployment via native Git integration.

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
If using Vercel Git integration, commit the refreshed `site/` folder and push to `main`.

## 5. Optional: deploy specific snapshot

```bash
python scripts/export_static_site.py \
  --snapshot-dir reports/locks/<snapshot_id> \
  --out-dir site
```
