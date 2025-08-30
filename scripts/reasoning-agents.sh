#!/bin/bash
# Automated Agent System for Guile Reasoning Model
# Creates specialized agents for different project components

# Define agent tasks based on our project structure
AGENT_TASKS=(
    "reasoning-core:Continue implementing core reasoning chain functionality in src/reasoning/core.scm. Focus on chain-of-thought patterns and problem decomposition."
    "reasoning-inference:Enhance inference-time scaling techniques in src/reasoning/inference.scm. Work on self-consistency, beam search, and Monte Carlo methods."
    "text-generation:Complete tokenizer and text generation in src/generation/text-gen.scm. Add special reasoning tokens and sampling strategies."
    "evaluation-metrics:Expand evaluation framework in experiments/05_evaluation_metrics.py. Add more sophisticated metrics and benchmark problems."
    "ollama-integration:Improve Ollama integration in experiments/04_ollama_reasoning.py. Add streaming support and model comparison features."
)

# Create tmux sessions for each agent
create_agent_sessions() {
    echo "🤖 Creating reasoning model agent sessions..."
    
    for task in "${AGENT_TASKS[@]}"; do
        IFS=":" read -r session command <<< "$task"
        
        if ! tmux has-session -t "$session" 2>/dev/null; then
            echo "Creating session: $session"
            tmux new-session -d -s "$session"
            tmux rename-window -t "$session:0" "work"
            
            # Set working directory to project root
            tmux send-keys -t "$session:work" "cd /home/dsp-dr/ghq/github.com/dsp-dr/guile-reasoning-model" C-m
            tmux send-keys -t "$session:work" "echo 'Agent $session ready. Waiting for commands...'" C-m
        else
            echo "Session $session already exists"
        fi
    done
}

# Agent coordination loop
run_agent_loop() {
    echo "🚀 Starting reasoning model agent coordination loop..."
    
    while true; do
        for task in "${AGENT_TASKS[@]}"; do
            IFS=":" read -r session command <<< "$task"
            if tmux has-session -t "$session" 2>/dev/null; then
                echo "[$(date +%H:%M:%S)] Sending task to $session"
                tmux send-keys -t "$session:work" "$command" C-m
                sleep 5  # Give each agent time to work
            fi
        done
        
        # Auto-commit progress every cycle
        if [ -n "$(git status --porcelain)" ]; then
            git add -A 2>/dev/null
            git commit -m "feat: reasoning model agent progress $(date +%H:%M)" 2>/dev/null
            echo "[$(date +%H:%M:%S)] Auto-committed changes"
        fi
        
        echo "[$(date +%H:%M:%S)] Cycle complete. Waiting 5 minutes..."
        sleep 300  # Every 5 minutes
    done
}

# Monitor agent status
show_agent_status() {
    echo "📊 Reasoning Model Agent Status:"
    echo "================================"
    
    for task in "${AGENT_TASKS[@]}"; do
        IFS=":" read -r session command <<< "$task"
        if tmux has-session -t "$session" 2>/dev/null; then
            echo "✅ $session: Active"
        else
            echo "❌ $session: Inactive"
        fi
    done
    
    # Show recent commits
    echo -e "\n📝 Recent Agent Commits:"
    git log --oneline --grep="agent progress" -5 2>/dev/null || echo "No agent commits yet"
}

# Stop all agents
stop_agents() {
    echo "🛑 Stopping reasoning model agents..."
    
    # Kill the background loop if running
    pkill -f "reasoning-agents.sh.*run_agent_loop"
    
    # Close tmux sessions
    for task in "${AGENT_TASKS[@]}"; do
        IFS=":" read -r session command <<< "$task"
        if tmux has-session -t "$session" 2>/dev/null; then
            tmux kill-session -t "$session"
            echo "Stopped session: $session"
        fi
    done
}

# View specific agent output
view_agent() {
    local agent=$1
    if [ -z "$agent" ]; then
        echo "Usage: $0 view <agent-name>"
        echo "Available agents:"
        for task in "${AGENT_TASKS[@]}"; do
            IFS=":" read -r session command <<< "$task"
            echo "  - $session"
        done
        return 1
    fi
    
    if tmux has-session -t "$agent" 2>/dev/null; then
        tmux capture-pane -t "$agent:work" -p
    else
        echo "Agent $agent not found or inactive"
    fi
}

# Main command dispatcher
case "$1" in
    "create")
        create_agent_sessions
        ;;
    "start")
        create_agent_sessions
        echo "Starting background agent loop..."
        nohup bash "$0" run_agent_loop > /tmp/reasoning-agents.log 2>&1 &
        echo "Agents started in background. Check status with: $0 status"
        echo "View logs: tail -f /tmp/reasoning-agents.log"
        ;;
    "run_agent_loop")
        run_agent_loop
        ;;
    "status")
        show_agent_status
        ;;
    "stop")
        stop_agents
        ;;
    "view")
        view_agent "$2"
        ;;
    "logs")
        tail -f /tmp/reasoning-agents.log 2>/dev/null || echo "No logs yet. Start agents with: $0 start"
        ;;
    *)
        echo "🧠 Guile Reasoning Model Agent System"
        echo "======================================"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  create  - Create agent tmux sessions"
        echo "  start   - Start automated agent system in background"
        echo "  status  - Show agent status and recent commits"
        echo "  stop    - Stop all agents and sessions"
        echo "  view    - View output from specific agent"
        echo "  logs    - Follow agent system logs"
        echo ""
        echo "Agents:"
        for task in "${AGENT_TASKS[@]}"; do
            IFS=":" read -r session command <<< "$task"
            echo "  - $session"
        done
        ;;
esac