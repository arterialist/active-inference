# DESIGN.md - PAULA Research Lab

## Context (from discovery)

- Artifact type: technical research instrument / dashboard
- Positioning: utilitarian, technical, research-grade
- Audience: the embodied-agent researcher inspecting neural and physical causality
- Primary action: select a live brain, move through exact ticks, and explain a behavior
- Adjectives: precise, inspectable, calm, tactile
- Visual word translations: precise -> visible grid rails and tabular numerals; inspectable -> every state has a source tick and endpoint; calm -> near-black surfaces and restrained color; tactile -> clear transport controls with immediate state feedback
- Aesthetic essence (3 words): instrument, evidence, control
- Single-minded proposition: the lab keeps the body, circuit, and causal evidence in one working view
- Mode: dark | Density: dense but zoned
- Constraints: plain HTML/CSS/JS, local HTTP APIs, WCAG AA focus and keyboard behavior, no framework build step

## Aesthetic

- Direction: field instrument / Swiss technical console
- Defining trait: a persistent evidence rail runs beside the 3D microscope, so a selected tick is never detached from its physical context
- Signature move: amber tick cursor and matching amber body/neural event markers across the timeline, body card, and inspector

## Typography

- Display: IBM Plex Sans (with Avenir Next fallback) | body: IBM Plex Sans | mono: IBM Plex Mono
- Scale: 1.25 Major Third, base 16px; display 24px, title 18px, section 14px, body 14px, meta 12px
- Weights: 400/500/600/700 | numerals use tabular figures

## Color

- Strategy: blue-green near-black surfaces keep the long microscope session quiet; amber is reserved for the current tick and controls; mint and coral carry semantic state with text labels
- Distribution: 65 neutral / 25 brand / 10 accent
- Palette: bg `oklch(15% 0.025 190)`; surface `oklch(20% 0.03 190)`; raised `oklch(24% 0.035 190)`; fg `oklch(94% 0.02 190)`; muted `oklch(68% 0.035 190)`; border `oklch(34% 0.035 190)`; accent amber `oklch(78% 0.15 75)`; success mint `oklch(76% 0.13 155)`; warning `oklch(80% 0.14 85)`; error coral `oklch(70% 0.16 28)`
- Dark mode: authored as the primary mode, with no pure black or glow shadows

## Spacing, radius, shadow

- Spacing base: 4px, scale 4/8/12/16/24/32
- Radius: 4px for controls, 8px for zones
- Shadow approach: defined edges only; elevation comes from surface lightness

## Layout and composition

- Grid: asymmetric operator console; 12-column desktop logic with a wide microscope and narrow evidence rail
- Signature layout move: the evidence rail remains visible while the microscope and harness dock change modes
- Density: dense | Scanning: F-pattern from command bar to tick rail to inspector
- Responsive: desktop-first; below 980px the rail stacks below the microscope, below 680px controls become a two-row transport bar

## Components and states

- Buttons: one amber primary transport action; outlined secondary controls; text tertiary links; hover/active/focus/disabled/loading states
- Inputs: visible labels, URL validation, inline connection errors, no placeholder-only labels
- Lists: light row separators, tabular numerals, keyboard-focusable selected rows
- Empty/loading/error: every panel has a useful next action and preserves its layout while loading
- Focus ring: 2px amber box-shadow with 2px offset

## Motion

- Duration scale: 120ms and 180ms; ease-out only
- Animate: opacity/transform for status changes; never animate layout or high-frequency tick updates
- Reduced motion: no transform transitions, direct state swaps

## Accessibility

- Native buttons, links, labels, select, range, and output elements
- Visible focus and keyboard-operable transport, tabs, timeline, and run cards
- Status uses text plus shape/labels, not color alone
- `aria-live` for connection and run updates; touch targets are at least 40px

## Cards and surfaces

- Zones use a surface background and defined border; avoid nested card stacks
- The inspector is a single rail with section rules rather than many floating cards

## Slop audit

- The lab avoids centered marketing layouts, indigo gradients, card grids as the only structure, diffuse shadows, and decorative motion.
- The distinctive choice is an operator console with a persistent evidence rail and exact tick cursor.

## Changelog

- 2026-08-04: Reframed the shell as a live causal inspection console with transport, tick timeline, body/neural inspector, and harness history.
