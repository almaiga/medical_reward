#!/bin/bash

# Start selfplay training in a screen session

SCREEN_NAME="selfplay_last"

# Check if screen session already exists
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "⚠️  Screen session '$SCREEN_NAME' already exists!"
    echo "Options:"
    echo "  1. Attach to existing session: screen -r $SCREEN_NAME"
    echo "  2. Kill existing session: screen -S $SCREEN_NAME -X quit"
    echo "  3. Use a different name by editing this script"
    exit 1
fi

echo "🚀 Starting selfplay training in screen session: $SCREEN_NAME"
echo ""
echo "To attach to the session later, run:"
echo "  screen -r $SCREEN_NAME"
echo ""
echo "To detach from the session (once attached):"
echo "  Press Ctrl+A, then D"
echo ""

# Start screen session and run training
screen -dmS "$SCREEN_NAME" bash -c "cd /workspace/medical_reward && ./run_selfplay_training.sh"

# Wait a moment for screen to start
sleep 2

# Check if it started successfully
if screen -list | grep -q "$SCREEN_NAME"; then
    echo "✅ Screen session '$SCREEN_NAME' started successfully!"
    echo ""
    echo "View the session with: screen -r $SCREEN_NAME"
else
    echo "❌ Failed to start screen session"
    exit 1
fi
