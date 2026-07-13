#!/usr/bin/env python3
"""Demo: Snowflake ID precision loss between order and ledger services."""
import requests
import json

ORDER_SERVICE = "http://localhost:8002"
LEDGER_SERVICE = "http://localhost:8003"

# Large Snowflake ID that exceeds JS safe integer limit
TXN_ID = 334596396910907392

print(f"=== Transaction ID Precision Loss Demo ===\n")
print(f"Original transaction_id: {TXN_ID}")
print(f"Type: {type(TXN_ID).__name__}")
print(f"JS safe integer limit:   {2**53 - 1}")
print(f"Exceeds safe limit:      {TXN_ID > 2**53 - 1}\n")

# 1. Create order
print("--- Step 1: POST /orders ---")
payload = {"transaction_id": TXN_ID, "amount": 199.99, "merchant": "demo"}
print(f"Request payload: {json.dumps(payload)}")

resp = requests.post(f"{ORDER_SERVICE}/orders", json=payload)
result = resp.json()
print(f"Response: {json.dumps(result, indent=2)}")

returned_id = result.get("transaction_id")
print(f"\nOriginal ID sent:     {TXN_ID}")
print(f"ID returned by order: {returned_id}")
print(f"IDs match:            {TXN_ID == returned_id}")

if TXN_ID != returned_id:
    print(f"\n>>> BUG: ID was silently rounded from {TXN_ID} to {returned_id}")

# 2. Try to fetch from ledger using the ORIGINAL ID
print(f"\n--- Step 2: GET /transactions/{TXN_ID} (original ID) ---")
resp = requests.get(f"{LEDGER_SERVICE}/transactions/{TXN_ID}")
print(f"Status: {resp.status_code}")
print(f"Response: {resp.json()}")

# 3. Try to fetch using the ROUNDED ID
print(f"\n--- Step 3: GET /transactions/{returned_id} (rounded ID) ---")
resp = requests.get(f"{LEDGER_SERVICE}/transactions/{returned_id}")
print(f"Status: {resp.status_code}")
print(f"Response: {resp.json()}")

print(f"\n=== Conclusion ===")
print(f"Transaction with original ID {TXN_ID} is NOT FOUND")
print(f"because JSON.parse() rounded it to {returned_id}")
print(f"The bug is in demo-order-service/src/order.ts — Express body parser uses JSON.parse()")
