# Credit Decision Engine — Zoot Rules Flow

## Overview

The Credit Decision Engine (`decision_engine.py`) applies **Zoot business rules** to customer credit applications. It integrates with the Vector Store for rule retrieval and request storage, producing decisions in the standard Zoot response format.

---

## Decision Flow Diagram

```mermaid
flowchart TD
    START([Customer Request Received]) --> LOAD[Load Customer Data from JSON files]
    LOAD --> INGEST[Ingest Request into ChromaDB Vector Store]
    INGEST --> CHUNK[Chunk request by section: applicant, financials, bureauData, etc.]
    CHUNK --> RETRIEVE[Retrieve matching rules from Vector Store via similarity search]

    RETRIEVE --> EXTRACT[Extract key fields]
    EXTRACT --> BS[Bureau Score]
    EXTRACT --> DTI[Calculate DTI Ratio = existingObligations / annualIncome]
    EXTRACT --> AMOUNT[Requested Amount & Tenure]

    BS --> RULE1{Bureau Score >= 700?}
    RULE1 -->|Yes| DTICHK{DTI <= 50%?}
    DTICHK -->|Yes| APPROVED[✅ APPROVED]
    DTICHK -->|No| REFER1[⚠️ REFER]

    RULE1 -->|No| RULE2{Bureau Score 620-699?}
    RULE2 -->|Yes| REFER2[⚠️ REFER]
    RULE2 -->|No| DECLINED[❌ DECLINED]

    APPROVED --> CALC[Calculate Credit Terms]
    CALC --> RATE[Interest Rate based on bureau score + DTI]
    CALC --> LIMIT[Credit Limit based on income - obligations]

    APPROVED --> RESP_A[Response: LOW risk, Z001 + Z014 reason codes]
    REFER1 --> RESP_R[Response: MEDIUM risk, Z023 + Z062 reason codes]
    REFER2 --> RESP_R2[Response: MEDIUM risk, Z023 + Z045 reason codes]
    DECLINED --> RESP_D[Response: HIGH risk, Z071 + Z062 reason codes]

    RESP_A --> TRACE[Add Traceability: ingestion info + matched rules + rules source]
    RESP_R --> TRACE
    RESP_R2 --> TRACE
    RESP_D --> TRACE
    RATE --> RESP_A
    LIMIT --> RESP_A

    TRACE --> RETURN([Return Zoot-format Response])
```

---

## Business Rules (from ZootRules.xlsx)

| Rule | Bureau Score | DTI Check | Decision Code | Risk Band | Risk Score Range |
|------|-------------|-----------|---------------|-----------|-----------------|
| 1    | >= 700      | <= 50%    | **APPROVED**  | LOW       | 200 - 380       |
| 2    | 620 - 699   | Any       | **REFER**     | MEDIUM    | 500 - 655       |
| 3    | < 620       | Any       | **DECLINED**  | HIGH      | 750 - 1450      |

### DTI (Debt-to-Income) Calculation

```
DTI = existingObligations / annualIncome
Threshold = 0.50 (50%)
```

If DTI exceeds 50%, even a high bureau score customer gets **REFER** instead of **APPROVED**.

---

## Reason Codes

| Code | Description | Used In |
|------|-------------|---------|
| Z001 | Strong bureau score | APPROVED |
| Z014 | Income sufficient for requested amount | APPROVED |
| Z023 | Thin credit file | REFER |
| Z045 | Recent delinquency observed | REFER (when DTI is OK) |
| Z062 | High debt-to-income ratio | REFER / DECLINED |
| Z071 | Low bureau score | DECLINED |

---

## Credit Terms Calculation (APPROVED only)

### Interest Rate
- Based on bureau score tier and DTI ratio
- Lower bureau score → higher rate
- Higher DTI → rate penalty applied

### Credit Limit
- Based on `annualIncome - existingObligations`
- Capped relative to requested amount
- Higher income surplus → higher limit

---

## Sample Test Scenarios

### Scenario 1: APPROVED
```json
{
  "customerId": "CI98765432",
  "name": "Anita Kumar",
  "bureauScore": 768,
  "annualIncome": 900000,
  "existingObligations": 150000,
  "requestedAmount": 300000,
  "DTI": "16.7% (within threshold)"
}
```
**Result:** APPROVED, Risk Score ~312, Risk Band LOW

### Scenario 2: DECLINED
```json
{
  "customerId": "CI98765444",
  "name": "Test Zoot",
  "bureauScore": 76,
  "annualIncome": 900000,
  "existingObligations": 150000,
  "DTI": "16.7% (within threshold)"
}
```
**Result:** DECLINED, Risk Score ~845, Risk Band HIGH (bureau score far below 620)

### Scenario 3: REFER
```json
{
  "customerId": "CI98765555",
  "name": "Test Zoot",
  "bureauScore": 630,
  "annualIncome": 900000,
  "existingObligations": 150000,
  "DTI": "16.7% (within threshold)"
}
```
**Result:** REFER, Risk Score ~625, Risk Band MEDIUM (bureau score in 620-699 range)

---

## Vector Store Integration

```mermaid
sequenceDiagram
    participant User
    participant QueryRouter as Smart Query Router
    participant Engine as CreditDecisionEngine
    participant VectorDB as ChromaDB

    User->>QueryRouter: "What is credit decision for CI98765432?"

    QueryRouter->>QueryRouter: Detect credit keywords ✓
    QueryRouter->>QueryRouter: Extract customer ID: CI98765432

    QueryRouter->>Engine: evaluate("CI98765432")

    Engine->>Engine: Load customer data from memory
    Engine->>VectorDB: Ingest request (chunked by section)
    VectorDB-->>Engine: doc_ids

    Engine->>VectorDB: Similarity search for matching rules
    VectorDB-->>Engine: Matched rule chunks

    Engine->>Engine: Apply rules (bureau=768, DTI=16.7%)
    Engine->>Engine: Decision: APPROVED, LOW risk

    Engine-->>QueryRouter: Response + Traceability
    QueryRouter-->>User: Decision card with details
```

---

## Response Format (Zoot Standard)

```json
{
  "header": {
    "requestId": "REQ-2025-0042",
    "responseTimestamp": "2026-04-05T10:30:00Z"
  },
  "decision": {
    "decisionCode": "APPROVED",
    "decisionDescription": "Approved",
    "creditLimit": 500000,
    "interestRate": 11.5,
    "tenureMonths": 24
  },
  "risk": {
    "riskScore": 312,
    "riskBand": "LOW"
  },
  "reasonCodes": [
    { "code": "Z001", "description": "Strong bureau score" },
    { "code": "Z014", "description": "Income sufficient for requested amount" }
  ]
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST   | `/query` | Smart query — auto-routes credit questions to decision engine |
| POST   | `/api/decision/evaluate/<customer_id>` | Evaluate a specific customer |
| POST   | `/api/decision/evaluate-request` | Evaluate an ad-hoc request payload |
| GET    | `/api/decision/customers` | List available customer IDs |
| POST   | `/api/decision/reload` | Reload customer data from disk |

---

## Entry Points

1. **Web UI** — Type a credit decision question in the query box (e.g., "credit decision for CI98765432")
2. **REST API** — Call `/api/decision/evaluate/CI98765432` directly
3. **Test Script** — Run `python ingest_zoot.py` to test all 3 scenarios (APPROVED, DECLINED, REFER)
