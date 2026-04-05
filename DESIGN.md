# Design System — Zoom AI Caption Overlay

## Product Context
- **What this is:** Real-time caption overlay for Zoom AI virtual camera output
- **Who it's for:** Meeting participants who need live captions with speaker identification
- **Space/industry:** AI meeting assistant, accessibility tools
- **Project type:** Virtual camera overlay (PIL/OpenCV rendering)

## Aesthetic Direction
- **Direction:** Modern/Friendly — Chat-app energy optimized for video meeting contexts
- **Decoration level:** Intentional — Soft shadows, rounded corners, enough polish to feel premium without clutter
- **Mood:** Approachable, professional, comfortable for long meetings. The overlay should feel like a helpful assistant, not technical equipment.
- **Reference:** Contemporary chat apps (Discord, Slack) meet meeting accessibility tools

## Typography

### Font Stack (Priority Order)
**Chinese:**
1. **PingFang SC** (macOS) — Primary choice. Modern, clean, excellent legibility at small sizes.
2. **Noto Sans CJK** (Linux) — Open source fallback. Widely available, consistent rendering.
3. **WQY Zenhei** (Linux) — Legacy fallback for older systems.

**English/Fallback:**
- System sans-serif default

### Font Sizes
- **Caption text:** 36px — Readable on typical meeting displays (1920x1080)
- **Speaker labels:** 28px — Secondary hierarchy, distinguishes speaker from content

### Loading Strategy
The `ChineseFontLoader` class implements a fallback chain:
1. Try PingFang SC first (modern, clean)
2. Fall back to Noto Sans CJK (open source)
3. Fall back to WQY Zenhei (legacy)
4. Use PIL default font as last resort

## Color

### Approach
Balanced — Primary + secondary colors for speaker differentiation, semantic colors for status.

### Palette (BGR for PIL/OpenCV)

**Background:**
- `background_color: (24, 24, 28)` — Softer dark, closer to modern apps like Discord/Slack
- `background_alpha: 0.90` — More opaque for better readability in varying lighting

**Text:**
- `text_color: (250, 250, 249)` — Warm white, not harsh pure white

**Speaker Identification (6 distinct colors):**
| Speaker | Color (BGR) | Hex | Name |
|---------|-------------|-----|------|
| 1 | `(59, 130, 246)` | #3B82F6 | Blue |
| 2 | `(139, 92, 246)` | #8B5CF6 | Purple |
| 3 | `(16, 185, 129)` | #10B981 | Green |
| 4 | `(245, 158, 11)` | #F59E0B | Orange |
| 5 | `(236, 72, 153)` | #EC4899 | Pink |
| 6 | `(20, 184, 166)` | #14B8A6 | Teal |

**Rationale:**
- Avoided red (can feel alarming/error-like in meetings)
- Cool tones (blue, purple, teal) dominate for professional feel
- Warm accent (orange) provides contrast
- All colors tested for adequate contrast on dark background

## Spacing

### Base Unit
4px (standard for digital design)

### Density
Comfortable — Generous padding for readability during extended meetings

### Scale
- `padding: 24px` — Internal padding for caption cards (was 20px)
- `margin_sides: 30px` — Side margins for viewport edge
- `line_height: 52px` — Line spacing for multi-line captions (was 55px, tighter for readability)

## Layout

### Approach
Grid-disciplined — Structured, predictable caption positioning

### Grid
- Single column, bottom-positioned by default
- Configurable: top, bottom, center

### Max Content Width
`width - (2 * margin_sides)` — Full width minus side margins

### Border Radius Scale
- Caption cards: `16px` — Friendlier, more rounded (was 15px)
- Speaker badges: `8px` — Smaller radius for secondary elements

## Motion

### Approach
Intentional — Smooth entrance animations that aid comprehension without distraction

### Easing
- Enter: Linear fade + linear slide (simple, predictable)
- Exit: N/A (captions persist, replaced by new content)

### Duration Scale
- `fade_in_duration: 0.25s` — Snappier, more premium (was 0.3s)
- `slide_offset: 40px` — Subtle slide distance (was 50px)

### Animation Types
1. **Fade in** — Opacity 0 → 1.0 over 0.25s
2. **Slide up** — Y offset 40px → 0 over fade duration

## Effects

### Shadows
- `shadow_blur: 25` — Soft, diffuse shadow (was 20)
- `shadow_offset: (0, 6)` — More depth, subtle lift effect (was (0, 4))
- Shadow color: `(0, 0, 0, 80)` — Semi-transparent black

### Transparency
- Background card: 90% opacity (0.90) — Readable but not solid
- Speaker badges: 70% opacity (180/255) — Secondary visual weight

## Caption Styles

The overlay supports 4 distinct styles for different meeting contexts:

### 1. Modern (Default)
- Floating card at bottom of frame
- Speaker name badges with color coding
- Rounded corners, soft shadows
- Best for: Professional meetings, presentations

### 2. Chat
- Chat bubble style, stacked upward from bottom
- Colored bubbles per speaker
- Tighter spacing
- Best for: Casual discussions, team standups

### 3. Karaoke
- Centered, large text
- Single caption visible at a time
- Speaker name above caption
- Best for: Speeches, performances, lectures

### 4. Subtitle
- Traditional subtitle bar at bottom
- No speaker badges (text only)
- Minimal styling
- Best for: Generic captioning, accessibility focus

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-04 | Initial design system created | Created by /design-consultation based on modern/friendly aesthetic for virtual camera overlay |
| 2026-04-04 | Switch from dark blue-gray (20,20,30) to softer dark (24,24,28) | Warmer tone more comfortable for extended meetings; closer to contemporary apps |
| 2026-04-04 | Redesigned speaker colors (Blue/Purple/Green/Orange/Pink/Teal) | Removed red (feels alarming), added teal for better differentiation; cool tones for professional feel |
| 2026-04-04 | Increased padding to 24px | More breathing room improves readability during long meetings |
| 2026-04-04 | Prioritized PingFang SC for Chinese text | Modern, clean font with excellent legibility; standard on macOS |
| 2026-04-04 | Snappier animations (0.25s fade, 40px slide) | More premium feel; less distracting than slower animations |

## Implementation Notes

### File: `zoom_ai/wlk_enhanced_overlay.py`

Key classes:
- `EnhancedOverlayConfig` — Dataclass containing all design system values
- `ChineseFontLoader` — Font loading with fallback chain
- `EnhancedCaptionRenderer` — Main renderer implementing 4 caption styles

### Updating the Design System

To modify colors, spacing, or effects:

```python
config = EnhancedOverlayConfig(
    background_color=(24, 24, 28),  # BGR
    padding=24,
    rounded_corners=16,
    # ... other values
)
renderer = EnhancedCaptionRenderer(config)
```

### Adding New Caption Styles

1. Add to `CaptionStyle` enum
2. Implement `_render_{style_name}` method in `EnhancedCaptionRenderer`
3. Add style to `render()` method's if/elif chain

## Accessibility Considerations

- **Color contrast:** All speaker colors tested for WCAG AA compliance on dark background
- **Font size:** 36px captions readable at typical viewing distances
- **Speaker identification:** Color coding + text labels for dual coding (helps colorblind users)
- **Animation:** Subtle, non-distracting; can be disabled via `AnimationConfig`

## Future Enhancements

Potential improvements for future iterations:

1. **Light mode option** — For brighter meeting environments
2. **Custom color themes** — User-selectable palettes
3. **Font size scaling** — Per-user preferences
4. **Animation speed control** — Accessibility option
5. **Colorblind-friendly mode** — High-contrast patterns + icons
