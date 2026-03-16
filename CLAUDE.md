# MLEvolve

## Dashboard

- Live dashboard: `python3 dashboard.py` (Flask, reads from `runs/` directory)
- Static dashboard deployed at: https://mlevolve-dashboard.vercel.app
- After a run completes, redeploy the static dashboard with: `./deploy_dashboard.sh`
  - This runs `export_dashboard.py` (exports all runs as static JSON to `dist/`) then `npx vercel --prod`
