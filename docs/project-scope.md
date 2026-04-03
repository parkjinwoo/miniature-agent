# Project Scope

`miniature-agent` is a small coding agent project for learning.

The goal is not to build the largest or most feature-rich agent. The goal is to build one that stays understandable while teaching useful lessons about agent design, memory, tools, sessions, and recovery.

## Direction

- keep the system small
- keep the behavior explicit
- keep the defaults stable
- keep the codebase readable for the next contributor

## What Matters

- stability
- soundness
- simplicity
- friendliness
- memory that can be resumed and used, not just stored

## What This Project Avoids

- feature growth for its own sake
- platform ambitions
- large extension systems
- new modes or subsystems when prompt, tools, or sessions are enough

## Rule Of Thumb

When a new idea appears, prefer:

1. improving an existing tool
2. improving session handling
3. improving clarity in the prompt or UI
4. not adding the feature

The project may change in later versions. This document only describes the current direction.
