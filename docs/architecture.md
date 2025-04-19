# Cognitive Architecture Overview

This document summarizes the **module layout** and communication graph (UML). For deep details, see the canvas docs.

| Module | Responsibility | Key Topics |
|--------|----------------|-----------|
| Perception | Sensor ingestion, object detection | `sensory_data.*` |
| Global Workspace | Context fusion & broadcast | `broadcasted_context` |
| Higher‑Order Thought | Confidence calc, introspection | `meta_state` |
| Attention Schema | Saliency & focus | `attention_focus` |
| Motivation | Drives → goals | `goals`, `drive_signals` |
| Language Interface | Self‑report via LLM | `verbal_output` |

![uml](../uml_diagram.png)