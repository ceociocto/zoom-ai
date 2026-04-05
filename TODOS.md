# TODOs

## Design

### Create DESIGN.md

**What:** Create DESIGN.md with color system, typography, component patterns for Echo UI.

**Why:** Establishes design tokens (color, typography, spacing) for all Echo UI components. Ensures consistency across efficiency reports, future UI additions, and visual feedback patterns.

**Pros:** Consistent visual language, faster implementation decisions, reusable components, on-brand experience.

**Cons:** Requires upfront time investment (~30-60 minutes with /design-consultation).

**Context:** Run `/design-consultation` before Phase 1 implementation. The skill will generate DESIGN.md based on product requirements and brand preferences. This is a prerequisite for polished, consistent UI across all Echo components.

**Depends on / blocked by:** None (can be done anytime before Phase 1 implementation)

---

### Empty State Illustration for "No Decisions Found"

**What:** Design or source an illustration for the empty state when no decisions are captured in a meeting.

**Why:** The first meeting with this system might have no decisions. The empty state sets the tone for the product — warm, supportive, not clinical. A well-designed empty state transforms "no data" into "good focused discussion."

**Pros:** Creates positive first impression, reduces perceived failure state, demonstrates thoughtfulness about user experience.

**Cons:** Requires illustration asset creation or licensing (can use simple geometric pattern or emoji-based design for MVP).

**Context:** Empty state message: "No decisions were captured in this meeting — good focused discussion! Decisions appear here when the group commits to action." Add to `echo/reporting/templates/efficiency_report.html` as the EMPTY state template.

**Depends on / blocked by:** DESIGN.md (to maintain visual consistency)

---

### Mobile Responsive Support for Efficiency Reports

**What:** Add mobile responsive breakpoints for efficiency reports after desktop MVP is validated.

**Why:** Desktop-only is appropriate for MVP, but if usage shows participants accessing reports from mobile devices, responsive design becomes important. Internal tool but users are increasingly mobile-first.

**Pros:** Accessible on all devices, future-proofs the product, supports "view report on the go" use case.

**Cons:** Additional design and implementation work, adds complexity to report template.

**Context:** Current plan specifies desktop-only (>1024px). This TODO tracks the need to add mobile (<768px) and tablet (768-1024px) breakpoints if usage data indicates demand. Add:
- Mobile: Single column, stacked metric cards
- Tablet: 2-column metric layout
- Touch-friendly sizing (44px min touch targets)

**Depends on / blocked by:** Usage data from Phase 1 deployment, DESIGN.md completion
