# Kalshi Trader - Deployment Guide

## Overview

This directory contains production deployment configurations for the Kalshi Trading Bot.

## Quick Start

### Docker Compose (Recommended for Single Node)

```bash
# 1. Setup environment
cp .env.production .env
# Edit .env with your credentials

# 2. Create secrets directory and add private key
mkdir -p secrets
cp /path/to/kalshi-key.pem secrets/
chmod 600 secrets/kalshi-key.pem

# 3. Start the trading bot
docker-compose up -d

# 4. View logs
docker-compose logs -f trader

# 5. Start with monitoring stack
docker-compose --profile monitoring up -d
```

### Kubernetes (Recommended for Production)

```bash
# 1. Create namespace and apply manifests
kubectl apply -f deploy/k8s/deployment.yaml

# 2. Create secrets (do NOT commit these to git!)
kubectl create secret generic kalshi-trader-secrets \
  --from-literal=KALSHI_API_KEY_ID=your_key_id \
  -n kalshi-trader

kubectl create secret generic kalshi-private-key \
  --from-file=kalshi-key.pem=/path/to/kalshi-key.pem \
  -n kalshi-trader

# 3. Check deployment status
kubectl get pods -n kalshi-trader
kubectl logs -f deployment/kalshi-trader -n kalshi-trader

# 4. Port forward for local access
kubectl port-forward svc/kalshi-trader 8080:8080 -n kalshi-trader
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Deployment Options                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     ┌──────────────────────────────────┐  │
│  │   Docker     │     │           Kubernetes             │  │
│  │   Compose    │     │                                  │  │
│  │              │     │  ┌──────────────────────────┐    │  │
│  │ ┌──────────┐ │     │  │      Deployment          │    │  │
│  │ │  Trader  │ │     │  │  ┌────────────────────┐  │    │  │
│  │ │  Service │ │     │  │  │   Trading Bot      │  │    │  │
│  │ └────┬─────┘ │     │  │  │  ┌──────────────┐  │  │    │  │
│  │      │       │     │  │  │  │  /health     │  │  │    │  │
│  │ ┌────┴─────┐ │     │  │  │  │  /ready      │  │  │    │  │
│  │ │Prometheus│ │     │  │  │  │  /metrics    │  │  │    │  │
│  │ │ Grafana  │ │     │  │  │  └──────────────┘  │  │    │  │
│  │ │  Loki    │ │     │  │  └────────────────────┘  │    │  │
│  │ └──────────┘ │     │  └──────────────────────────┘    │  │
│  └──────────────┘     └──────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Security

### Private Key Management

**IMPORTANT:** The private key is mounted as a file, NOT passed as an environment variable.

- Docker: Mounted from `./secrets/` directory (read-only)
- Kubernetes: Mounted from Secret as a file with `defaultMode: 0400`

### Secrets Rotation

A CronJob is included for key rotation (see `deploy/k8s/secrets.yaml`):

```bash
# Manually trigger rotation
kubectl create job --from=cronjob/kalshi-key-rotation key-rotation-manual -n kalshi-trader
```

For automated rotation, integrate with:
- AWS Secrets Manager + External Secrets Operator
- HashiCorp Vault
- Azure Key Vault

### Network Security

- **Docker:** Uses internal bridge network, ports only exposed for metrics
- **Kubernetes:** NetworkPolicy restricts egress to Kalshi API (443) and DNS only

## Monitoring

### Endpoints

| Endpoint | Purpose | Auth Required |
|----------|---------|---------------|
| `/health` | Liveness probe | No |
| `/ready` | Readiness probe | No |
| `/metrics` | Prometheus metrics | No |

### Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `trades_total` | Counter | Total trades by market/side/status |
| `latency_ms` | Histogram | API call latency |
| `errors_total` | Counter | Errors by type |
| `pnl_total` | Gauge | Total P&L |
| `win_rate` | Gauge | Win rate (0-1) |
| `positions_open` | Gauge | Current open positions |
| `trades_per_day` | Gauge | Trades in last 24h |
| `circuit_breaker_active` | Gauge | Circuit breaker status |

### Grafana Dashboard

Access at `http://localhost:3000` (default credentials: admin/admin)

Dashboard includes:
- Bot status overview
- Trade rate visualization
- P&L tracking
- Error monitoring
- API latency percentiles

## Health Checks

### Liveness Probe

Checks if the process is running. Returns 503 if the bot has stopped.

```bash
curl http://localhost:8080/health
```

### Readiness Probe

Checks if the bot is ready to trade. Returns 503 during initialization.

```bash
curl http://localhost:8080/ready
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs trader

# Verify secrets are mounted
docker-compose exec trader ls -la /secrets/

# Test with shell
docker-compose run --rm trader sh
```

### K8s Pod Pending

```bash
# Check events
kubectl get events -n kalshi-trader --sort-by='.lastTimestamp'

# Check PVC
kubectl get pvc -n kalshi-trader

# Describe pod
kubectl describe pod -l app=kalshi-trader -n kalshi-trader
```

### Metrics Not Showing

```bash
# Check metrics endpoint
curl http://localhost:8080/metrics

# Check Prometheus targets
kubectl port-forward svc/prometheus 9090:9090 -n kalshi-trader
# Open http://localhost:9090/targets
```

## Backup and Recovery

### Data Backup

```bash
# Docker
docker run --rm -v kalshi_trader_trader-data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup.tar.gz -C /data .

# Kubernetes
kubectl exec -n kalshi-trader deployment/kalshi-trader -- tar czf - /app/data > data-backup.tar.gz
```

### Disaster Recovery

1. Restore secrets from backup
2. Apply K8s manifests
3. Restore data volume
4. Verify health endpoints
5. Monitor for first hour

## Maintenance

### Updates

```bash
# Docker Compose
docker-compose pull
docker-compose up -d

# Kubernetes
kubectl set image deployment/kalshi-trader trader=kalshi-trader:new-tag -n kalshi-trader
kubectl rollout status deployment/kalshi-trader -n kalshi-trader
```

### Scaling

**Note:** Trading bot should remain a singleton (1 replica) to prevent duplicate trades.

For high availability, consider:
- Primary/standby with leader election
- Manual failover procedures
- State externalization (Redis/database)

## Support

For issues and questions:
1. Check logs: `docker-compose logs` or `kubectl logs`
2. Verify configuration: `docker-compose config`
3. Test connectivity: `curl http://localhost:8080/health`
