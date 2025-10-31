#!/bin/bash
# Quick start script for PhonoLex web app

echo "🚀 Starting PhonoLex Web App"
echo ""

# Check if in correct directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the webapp/ directory"
    exit 1
fi

# Start backend
echo "📡 Starting backend API (http://localhost:8000)..."
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to initialize
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Start frontend
echo "🎨 Starting frontend (http://localhost:3000)..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ PhonoLex is running!"
echo ""
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Trap Ctrl+C to kill both processes
trap "echo ''; echo '🛑 Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT

# Wait for either process to exit
wait
