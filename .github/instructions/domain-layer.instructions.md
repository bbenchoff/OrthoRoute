---
applyTo: "orthoroute/domain/**"
---

# Domain Layer Rules

The `domain/` layer is the **innermost layer** — it must have zero knowledge of infrastructure, frameworks, or KiCad.

## Hard Rules

- **No infrastructure imports.** Never import from `orthoroute.infrastructure.*` inside `domain/`.
- **No application imports.** `domain/` cannot import from `orthoroute.application.*`.
- **No KiCad bindings.** No `pcbnew`, `kiutils`, or any KiCad-specific type anywhere in this layer.
- **No I/O.** No file reads, network calls, or subprocess usage.

## Allowed Dependencies

```python
# ✅ Standard library + pure third-party only
import dataclasses, typing, enum, math
import numpy as np  # pure computation only

# ✅ Within domain
from orthoroute.domain.models.board import Board
from orthoroute.domain.services.routing_engine import RoutingEngine  # abstract base
```

## Patterns

- Models: `@dataclass(frozen=True)` for value objects; one class per file.
- Services: Abstract base classes (ABCs) only — concrete implementations live in `infrastructure/`.
- Events: Plain dataclasses, no event bus wiring (bus lives in `application/`).

## When you need external data

Define an interface in `application/interfaces/` and inject it via the constructor. Never reach out directly.
