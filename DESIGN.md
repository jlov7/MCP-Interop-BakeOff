# DESIGN.md

## 1. Visual Theme & Atmosphere

The interface is a signal cockpit for agent-tool reliability work. It feels like a late-night release console on a calibrated engineering monitor: dark, quiet, dense, and alive only where state changes matter. The UI should look credible to users who live in Linear, Sentry, Raycast, Vercel, and production observability tools.

Design register: product. Density: 7/10. Variance: 5/10. Motion: 4/10. Color strategy: restrained.

## 2. Color Palette & Roles

Use OKLCH tokens in CSS. Never use pure black or pure white.

- **Abyss Canvas** `oklch(12% 0.012 165)` — app background.
- **Carbon Rail** `oklch(16% 0.014 165)` — sidebars and persistent navigation.
- **Panel Surface** `oklch(19% 0.013 165)` — primary panels, tables, inspectors.
- **Panel Lift** `oklch(23% 0.014 165)` — selected rows and raised controls.
- **Warm Line** `oklch(31% 0.012 165)` — borders, dividers, grid lines.
- **Signal Green** `oklch(78% 0.16 158)` — primary action, success, current selection, live state.
- **Amber Warning** `oklch(78% 0.12 78)` — warning and approval attention.
- **Coral Failure** `oklch(68% 0.16 25)` — errors and failed regressions.
- **Ink Primary** `oklch(93% 0.006 165)` — main text.
- **Ink Secondary** `oklch(72% 0.008 165)` — body, secondary labels.
- **Ink Muted** `oklch(54% 0.007 165)` — metadata and inactive controls.

Signal Green is not a decorative fill. It appears on active states, focused controls, pass indicators, and the main run command.

## 3. Typography Rules

- **UI and Display**: Geist, `-apple-system`, BlinkMacSystemFont, `Segoe UI`, sans-serif.
- **Code and Numbers**: Geist Mono, `SFMono-Regular`, Menlo, Monaco, Consolas, monospace.
- Product labels and controls use fixed rem sizes, not viewport-scaled type.
- Numbers, timestamps, run IDs, latencies, and command text use mono.
- Body copy is capped at 75ch. Tables and traces may run wider.
- Letter spacing is zero except small mono labels may use `0.02em`.

## 4. Component Stylings

- **Buttons**: 6px radius, 1px line, 36-40px minimum height, clear hover/focus/active/disabled states. Primary uses Signal Green on dark text only when contrast passes; otherwise dark surface with Signal Green text.
- **Panels**: use flat surfaces, 1px borders, and small radius. Avoid nested cards. Tables, rails, timelines, and inspectors are preferred over repeated cards.
- **Tables and matrices**: dense row height, sticky headers where useful, mono numeric cells, semantic pass/warn/fail marks, selectable rows.
- **Trace rail**: vertical event stream with timestamps, tool names, latency, approval badges, and expandable structured payloads.
- **Loading**: skeleton rows that match table and panel geometry.
- **Empty states**: explain the next CLI action or run command, never just “No data.”
- **Errors**: inline with affected surface, include failing command or artifact path.

## 5. Layout Principles

- First route is the working console, not a landing page.
- Desktop layout: left navigation rail, top run command/status strip, central matrix, right inspector.
- Mobile layout: single column with collapsible navigation and inspector below the selected matrix.
- Use CSS Grid for app regions. Avoid flex percentage math.
- No horizontal overflow below 768px.
- Do not use identical 3-column feature-card grids.
- Cards are allowed only for repeated run summaries or compact artifacts.

## 6. Motion & Interaction

- Motion communicates state changes: run starting, matrix update, trace selection, copied command, artifact reveal.
- Animate transform and opacity only.
- Use 150-220ms ease-out transitions.
- Respect `prefers-reduced-motion`.
- Do not add decorative page-load choreography.

## 7. Anti-Patterns

- No emojis.
- No pure black or pure white.
- No gradient text.
- No neon glows.
- No glassmorphism as default.
- No generic “AI workflow” purple-blue palette.
- No hero metrics template.
- No repeated icon-card grids.
- No decorative badges or pills above main headings.
- No fake round numbers. Use organic demo values with traceable sample data.
- No filler phrases: elevate, unleash, next-gen, seamless.
