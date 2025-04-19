# Global Workspace Bus Schema

All messages are JSON with `timestamp` (UTC ms) + payload fields.

```json
{
  "timestamp": 1713550000,
  "entities": [
    {
      "id": "obj‑1337",
      "class": "cup",
      "position": [1.2, 0.4, 0.8],
      "confidence": 0.92
    }
  ]
}
