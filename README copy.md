# SaaS Agent Team

Multi-agent AI system for autonomous SaaS development with cost-optimized LLM routing.

## Architecture

```
saas-agent-team/
├── core/           # LLM router, cost tracker, cache manager, batch processor
├── agents/         # 7 specialized agents (Architect, PM, Dev, QA, Security, DevOps, Research)
├── orchestration/  # Workflow graph, crew manager, task queue, human-in-loop
├── memory/         # Vector store (Weaviate), cache (Redis), database (PostgreSQL), RAG
├── api/            # FastAPI REST API with WebSocket support
├── observability/  # Structured logging, Prometheus metrics, LangSmith tracing
└── tests/          # pytest test suite
```

## Cost Optimization

Four mechanisms work together to minimize LLM costs:

| Mechanism                | Savings     | How                                                                 |
| ------------------------ | ----------- | ------------------------------------------------------------------- |
| **LLM Routing**          | -60 to -80% | Each agent uses the optimal model for its task profile              |
| **Prompt Caching**       | -40 to -60% | Stable content (system prompts, project context) cached server-side |
| **Batch API**            | -50%        | Non-urgent tasks processed via batch endpoints                      |
| **Complexity Downgrade** | -70%        | Simple tasks auto-routed to cheaper models                          |

### Routing Table

| Agent     | Primary Model     | Fallback          | Rationale             |
| --------- | ----------------- | ----------------- | --------------------- |
| Architect | claude-opus-4     | claude-sonnet-4-5 | Maximum reasoning     |
| PM        | claude-sonnet-4-5 | gpt-4o            | Writing + specs       |
| Dev       | claude-sonnet-4-5 | gpt-4o            | Best code/price ratio |
| QA        | claude-haiku-4-5  | gpt-4o-mini       | Simple validation     |
| Security  | claude-sonnet-4-5 | gpt-4o            | Critical analysis     |
| DevOps    | claude-haiku-4-5  | gpt-4o-mini       | Templates + scripts   |
| Research  | gpt-4o            | claude-sonnet-4-5 | Web search capability |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- API keys: Anthropic, OpenAI

### Setup

```bash
# Clone and enter project

# Copy environment config
cp .env.example .env
# Edit .env with your API keys

# Start infrastructure
docker compose up -d

# Install Python dependencies
pip install -e ".[dev]"

# Run the API
uvicorn api.main:app --reload
```

### Running Tests

```bash
pytest tests/ -v
```

## API Endpoints

### Projects

- `POST /projects` -- Create a new project
- `GET /projects` -- List projects
- `GET /projects/{id}` -- Get project details
- `POST /projects/{id}/workflow` -- Start development workflow
- `GET /projects/{id}/status` -- Get workflow status

### Agents

- `POST /agents/execute` -- Execute a task with a specific agent
- `GET /agents/roles` -- List available agent roles

### Costs

- `GET /costs/summary` -- Cost summary (day/week/month)
- `GET /costs/{project_id}/breakdown` -- Cost breakdown by agent
- `GET /costs/{project_id}/savings` -- Cache + batch savings
- `GET /costs/{project_id}/projection` -- Monthly cost projection
- `POST /costs/{project_id}/budget` -- Set monthly budget
- `GET /costs/optimization-tips` -- Automated optimization suggestions
- `WebSocket /ws/costs/live` -- Real-time cost streaming

### Health

- `GET /health` -- Service health check
- `GET /metrics` -- Prometheus metrics

## Development Workflow

The system executes a multi-phase workflow:

```
budget_check → cache_warmup → research → design
→ [human approval] → development_loop → batch_collection
→ integration → deploy → [human approval] → cost_report
```

Each phase has budget checks. If budget is exceeded, all agents auto-downgrade to the cheapest model.

## Infrastructure

Docker Compose includes:

- **app** -- FastAPI application
- **worker** -- Celery worker for async tasks
- **beat** -- Celery beat for periodic jobs (batch polling, budget alerts)
- **redis** -- Cache + message broker
- **postgres** -- Persistent storage
- **weaviate** -- Vector database for RAG
- **prometheus** -- Metrics collection
- **grafana** -- Dashboards (http://localhost:3000, admin/admin)

## Configuration

All configuration via environment variables (see `.env.example`):

- `MONTHLY_BUDGET_USD` -- Maximum monthly LLM spend (default: $200)
- `DAILY_BUDGET_USD` -- Maximum daily spend (default: $10)
- `WARNING_THRESHOLD` -- Alert at this budget percentage (default: 70%)
- `ENABLE_PROMPT_CACHING` -- Enable Anthropic prompt caching (default: true)
- `ENABLE_BATCH_API` -- Enable batch processing (default: true)
- `ENABLE_COST_TRACKING` -- Enable cost tracking (default: true)

## License

MIT
