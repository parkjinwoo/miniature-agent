# Project Scope

`miniature-agent` is a small, durable, local-first coding agent.

The goal is not to build the largest or most feature-rich agent. The goal is to build one whose execution can be understood, whose history can be trusted, and whose work can be resumed after interruption.

## Direction

- keep the system small and inspectable
- treat the session format as durable user data
- make side effects and trust decisions explicit
- support multiple model providers through one tested behavior contract
- keep the codebase readable for the next contributor

## What Matters

- stability
- soundness
- simplicity
- friendliness
- memory that can be resumed and used, not just stored
- clear trust boundaries
- deterministic recovery
- provider conformance

## What This Project Avoids

- feature growth for its own sake
- platform ambitions
- large extension systems
- multi-agent orchestration
- unrestricted automation as a default
- new modes or subsystems when prompt, tools, or sessions are enough

## Rule Of Thumb

When a new idea appears, prefer:

1. improving an existing tool
2. improving session handling
3. improving clarity in the prompt or UI
4. not adding the feature

The phased implementation plan is in [`modernization-plan.md`](modernization-plan.md).
