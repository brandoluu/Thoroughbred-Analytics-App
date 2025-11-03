#!/bin/bash
# start.sh

set -e  # exit if any command fails

echo "🚧 Building frontend..."
cd frontend
npm install
npm run build
cd ..

echo "📦 Installing backend dependencies..."
pip install -r backend/requirements.txt

echo "📁 Moving frontend build to backend static folder..."
mkdir -p backend/static
cp -r frontend/dist/* backend/static/

echo "🚀 Starting FastAPI server..."
uvicorn backend.app:app --host 0.0.0.0 --port $PORT