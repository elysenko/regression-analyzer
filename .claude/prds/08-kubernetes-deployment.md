---
name: 08-kubernetes-deployment
description: Deploy regression-analyzer to Kubernetes
status: backlog
created: 2026-01-20T20:32:34Z
updated: 2026-01-20T20:32:34Z
---

# PRD: Kubernetes Deployment

## Overview

Deploy the containerized regression-analyzer API to a Kubernetes cluster with proper resource management, scaling, and monitoring.

## Problem Statement

The application needs:
- High availability deployment
- Automatic scaling based on load
- Secret management for API keys
- Persistent storage for uploads and outputs
- Service exposure for external access

## Requirements

### Functional Requirements

1. **Deployment**
   - 2 replicas minimum for HA
   - Rolling update strategy
   - Resource limits and requests
   - Liveness and readiness probes

2. **Service**
   - ClusterIP for internal access
   - NodePort or LoadBalancer for external access
   - Port 8000

3. **Secrets**
   - Store API keys as Kubernetes secrets
   - Mount as environment variables

4. **Storage**
   - PersistentVolumeClaim for /data (uploads)
   - PersistentVolumeClaim for /output (charts)
   - ReadWriteMany access mode for multi-replica

5. **ConfigMap**
   - Application configuration
   - Log level settings

### Non-Functional Requirements

- Zero-downtime deployments
- Auto-scaling: 2-10 replicas
- Health check timeout: 30s
- Storage: 10Gi data, 5Gi output

## Technical Approach

### Directory Structure

```
k8s/
├── namespace.yaml
├── secrets.yaml
├── configmap.yaml
├── pvc.yaml
├── deployment.yaml
├── service.yaml
└── hpa.yaml
```

### Manifests

#### namespace.yaml
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: regression-analyzer
```

#### deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: regression-analyzer
  namespace: regression-analyzer
spec:
  replicas: 2
  selector:
    matchLabels:
      app: regression-analyzer
  template:
    metadata:
      labels:
        app: regression-analyzer
    spec:
      containers:
      - name: api
        image: registry/regression-analyzer:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        envFrom:
        - secretRef:
            name: regression-analyzer-secrets
        - configMapRef:
            name: regression-analyzer-config
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: output
          mountPath: /app/output
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: regression-analyzer-data
      - name: output
        persistentVolumeClaim:
          claimName: regression-analyzer-output
```

#### service.yaml
```yaml
apiVersion: v1
kind: Service
metadata:
  name: regression-analyzer
  namespace: regression-analyzer
spec:
  type: NodePort
  selector:
    app: regression-analyzer
  ports:
  - port: 8000
    targetPort: 8000
    nodePort: 30080
```

#### hpa.yaml
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: regression-analyzer
  namespace: regression-analyzer
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: regression-analyzer
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Deployment Process

```bash
# Create namespace and resources
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Verify deployment
kubectl get pods -n regression-analyzer
kubectl get svc -n regression-analyzer
```

## Success Criteria

- All pods running and healthy
- Service accessible via NodePort
- Rolling updates work without downtime
- HPA scales based on CPU load
- Secrets properly mounted

## Dependencies

- PRD-06: FastAPI Web API
- PRD-07: Docker containerization
- Kubernetes cluster with storage provisioner

## Out of Scope

- Ingress controller configuration
- TLS/HTTPS termination
- Network policies
- Monitoring stack (Prometheus/Grafana)
