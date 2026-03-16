#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "Exporting static dashboard..."
python3 export_dashboard.py --output-dir dist
echo ""
echo "Deploying to Vercel..."
npx vercel --prod --yes
echo ""
echo "Done! https://mlevolve-dashboard.vercel.app"
