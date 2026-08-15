# UI Standard

## Purpose

This document defines the UI standard for Torch Houdini Node.

The interface should feel simple, precise, and production-friendly. The style is inspired by Google Material Design: clean layout, clear hierarchy, consistent spacing, calm color usage, and predictable interaction.

The UI should help users understand procedural node workflows, tensor data, model settings, and execution feedback without visual noise.

## Design Principles

### Clarity First

Every screen, panel, node, and control should make its purpose obvious.

Users should quickly understand:

- What this view is for.
- What can be edited.
- What is currently selected.
- What the system is doing.
- Whether something failed.

### Minimal But Not Empty

The interface should be visually simple, but not vague.

Prefer useful structure over decoration:

- Clear section titles.
- Consistent spacing.
- Lightweight dividers.
- Compact controls.
- Informative empty states.

Avoid decorative UI that does not help the workflow.

### Procedural Visibility

Because this project is node-based, the UI should always respect data flow.

Users should be able to see:

- Node input ports.
- Node output ports.
- Connected data paths.
- Node execution state.
- Tensor shape and dtype when relevant.

### Consistency Over Novelty

Use the same control style everywhere unless there is a strong reason not to.

Good UI here should feel learnable. Once the user understands one node panel, they should understand the next one.

## Visual Style

## Layout

Use a clean grid system.

Recommended layout rules:

- Use 8px as the base spacing unit.
- Prefer spacing values such as 8, 16, 24, 32, and 48.
- Align labels, inputs, and values consistently.
- Avoid floating decorative cards.
- Use panels only when they represent real tools, inspectors, or repeated items.

## Color

Use a restrained color palette.

Recommended roles:

- Background: neutral light or neutral dark.
- Surface: slightly elevated panel color.
- Primary: one clear accent color for important actions.
- Secondary: reserved for supporting actions.
- Error: clear red tone.
- Warning: amber tone.
- Success: green tone.
- Info: blue tone.

Do not overuse accent colors. Most of the interface should remain neutral.

## Typography

Typography should be quiet and readable.

Recommended rules:

- Use one primary UI font family.
- Use clear size hierarchy.
- Keep labels short.
- Use medium weight for section headers.
- Avoid oversized text inside technical panels.
- Do not use negative letter spacing.

Suggested hierarchy:

- Page title: 24px to 28px.
- Section title: 16px to 18px.
- Body text: 14px to 16px.
- Metadata text: 12px to 13px.
- Button text: 14px.

## Shape

Use modest corner radius.

Recommended:

- Buttons: 6px to 8px.
- Panels: 8px.
- Input fields: 6px.
- Node boxes: 8px.

Avoid overly round, playful shapes. This tool should feel calm and technical.

## Elevation

Use elevation sparingly.

Prefer:

- Border.
- Background contrast.
- Small shadow only for overlays, menus, and active floating panels.

Avoid heavy shadows.

## Component Standards

## Buttons

Buttons should be used for clear actions.

Recommended button types:

- Primary button: main action, such as Run Graph.
- Secondary button: supporting action, such as Save Preset.
- Icon button: common tool action, such as reset, visibility, lock, or delete.
- Destructive button: delete or remove actions.

Button labels should be short and verb-based.

Good examples:

- Run
- Save
- Reset
- Export
- Add Node

Avoid vague labels:

- OK
- Do Thing
- Process
- Magic

## Inputs

Input fields should be predictable and compact.

Use:

- Text fields for names and paths.
- Number fields for scalar values.
- Sliders for bounded continuous values.
- Select menus for finite options.
- Toggles for binary settings.
- File pickers for model paths or asset paths.

Every input should have a label.

## Node Cards

Node cards represent computational units.

Each node card should show:

- Node name.
- Node type.
- Input ports.
- Output ports.
- Execution status.
- Error indicator when needed.

Node cards should not contain too much detail. Deeper configuration belongs in the inspector panel.

## Inspector Panel

The inspector panel shows details for the selected node.

Recommended sections:

- Summary.
- Parameters.
- Inputs.
- Outputs.
- Tensor Preview.
- Execution.
- Debug.

The inspector should be dense but not crowded.

## Graph Canvas

The graph canvas should prioritize readability.

Recommended behavior:

- Clear node positioning.
- Visible connections.
- Zoom and pan.
- Selection state.
- Error state.
- Execution state.

Connection lines should be subtle by default and stronger when selected.

## Status And Feedback

The UI must communicate state clearly.

Recommended states:

- Idle.
- Running.
- Success.
- Warning.
- Error.
- Disabled.
- Loading.

Errors should include:

- What failed.
- Where it failed.
- How to fix it when possible.

## Material Design Influence

This project should borrow the following Material Design ideas:

- Clear surfaces.
- Predictable controls.
- Consistent spacing.
- Strong hierarchy.
- Useful motion.
- Accessible color contrast.
- Direct manipulation.

But the project should not copy Material Design blindly. Houdini-style procedural work requires denser technical panels and graph-first workflows.

## Motion

Motion should explain state changes, not decorate the UI.

Use subtle transitions for:

- Panel open and close.
- Node selection.
- Execution progress.
- Error reveal.
- Menu appearance.

Avoid slow or dramatic animation.

## Accessibility

The UI should be usable in long technical sessions.

Requirements:

- High contrast text.
- Keyboard navigation where practical.
- Visible focus states.
- Do not rely on color alone for errors.
- Keep click targets large enough.
- Avoid tiny unlabeled controls.

## Naming And Copy

Use direct, technical language.

Prefer:

- Tensor Shape
- Input Dtype
- Run Graph
- Model Path
- Execution Device

Avoid:

- AI Magic
- Smartify
- Super Mode
- Process Stuff

## AI Agent UI Rule

When an AI coding agent implements UI, it must:

1. Read this document first.
2. Preserve visual consistency.
3. Keep controls predictable.
4. Avoid decorative complexity.
5. Add clear empty, loading, success, and error states.

## First UI Scope

The first UI prototype should include:

- Graph canvas.
- Node card style.
- Inspector panel.
- Parameter controls.
- Execution status.
- Tensor summary display.

Do not build a marketing landing page as the first screen.

The first screen should be the actual tool.
