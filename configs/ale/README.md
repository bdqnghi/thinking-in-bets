# Agents' Last Exam (ALE) — Container Configuration

Configuration files for running the container-based Agents' Last Exam (`rdi-berkeley/agents-last-exam`) experiments with Draft-First Betting.

## Setup

Clone the official benchmark repository alongside this repository:

```bash
git clone https://github.com/rdi-berkeley/agents-last-exam.git ../agents-last-exam
cd ../agents-last-exam
uv sync --all-packages
docker pull agentslastexam/ale-ubuntu22-docker:latest

cp ../thinking-in-bets/configs/ale/tib_ale_claw_S.yaml configs/agents/
cp ../thinking-in-bets/configs/ale/tib_docker.yaml      configs/environments/
cp ../thinking-in-bets/configs/ale/tib_smoke1.txt       selected_tasks/
cp ../thinking-in-bets/configs/ale/tib_smoke_exp.yaml   .
printf 'GEMINI_API_KEY=%s\n' "$GEMINI_API_KEY" > secret/.env

# Apply the Thinking-in-Bets Draft-First patch if running the Draft-First arm:
git apply ../thinking-in-bets/third_party/patches/ale_claw_tib.patch

uv run --no-sync python -m ale_run run tib_smoke_exp.yaml
```
