# Design System (DESIGN_SYSTEM.md)

## 1. Design Direction
- **Style:** Precision Visual Document Inspector & Engineering Intelligence Workspace, Dark-Mode First.
- **Mood:** Analytical, crystal-clear, evidence-driven, high-contrast.
- **Design Principles:**
  - *Visual Proof Next to Every Claim:* Whenever an answer references a chart, diagram, or table, the exact cropped region from the PDF is displayed inline beside the explanation.
  - *Modality Transparency:* Users immediately see whether the query was answered via the **⚡ Fast Text/Table Path** or the **👁️ Visual VLM Reasoning Path**, along with retrieval latency and confidence scores.
  - *Bilingual Technical Ergonomics:* Seamless RTL rendering for Persian analytical prose paired with strict LTR containers for Markdown tables, bounding-box coordinates, and technical metrics.

---

## 2. Brand Identity
- **Brand Name:** OmniDoc-Vision
- **Logo Badge:** Monospace emblem `[👁️ / OmniDoc-Vision]` in Violet/Cyan gradient on Deep Obsidian.
- **Brand Voice:** Objective, precise, citation-grounded.

---

## 3. Color Palette
- **Primary (Visual Intelligence Violet):** `#8B5CF6` (Violet 500 - VLM Visual Reasoning path & figure badges)
- **Secondary (Structured Table Cyan):** `#06B6D4` (Cyan 500 - Docling Table Reconstruction & Fast Path badges)
- **Accent (Verified Grounding Emerald):** `#10B981` (Emerald 500 - High CRAG confidence $\ge 0.75$)
- **Warning (Degraded/Noisy Scan Amber):** `#F59E0B` (Amber 500 - Medium CRAG confidence or OCR fallback warning)
- **Danger (Low Confidence / Rejected):** `#EF4444` (Red 500)
- **Background Primary (Dark):** `#090D16` (Deep Obsidian Slate)
- **Surface / Citation Card (Dark):** `#111827` (Gray 900)
- **Elevated Image Frame:** `#1E293B` (Slate 800 with `1px solid #334155` border)
- **Text Primary:** `#F8FAFC`
- **Text Secondary / Metadata:** `#94A3B8`

---

## 4. Typography
- **Primary Persian Font:** `Vazirmatn`, sans-serif (400, 500, 700)
- **Primary English Font:** `Inter`, system-ui, sans-serif
- **Monospace (Tables, Bounding Boxes, Metrics):** `JetBrains Mono`, `Fira Code`, monospace
- **Type Scale:**
  - **H1 (Document / Report Title):** 24px / 700 / Line Height 1.3
  - **H2 (Section Heading):** 18px / 600 / Line Height 1.4
  - **Body (Analytical Answer):** 15px / 400 / Line Height 1.7
  - **Table Cell Content:** 13.5px / 400 / Line Height 1.45
  - **Bounding Box & Telemetry Pills:** 12px / 500 / Monospace

---

## 5. Component Specifications

### 5.1 Adaptive Route & Telemetry Header Pill
Displayed at the top of every assistant response:
- **Fast Path Example:** `[⚡ Route: Fast Text/Table Path]  [⏱️ 218ms]  [🎯 Rerank Score: 0.94]` (Cyan border `#06B6D4`)
- **Visual Path Example:** `[👁️ Route: Visual VLM Reasoning (Qwen2.5-VL)]  [⏱️ 940ms]  [🖼️ 1 Figure Inspected]` (Violet border `#8B5CF6`)

### 5.2 Visual Grounding Card (`cl.Image` + Metadata Footer)
Whenever a `figure` or `table` crop is retrieved:
- **Image Container:** Dark slate frame (`#1E293B`), rounded `10px`, subtle zoom-on-click lightbox enabled in Chainlit (`display="inline"`, `size="large"`).
- **Attribution Footer:**
  - `📄 Document:` `cloud_database_benchmark_2026.pdf`
  - `📍 Location:` `Page 4  |  BBox: [0.19, 0.285, 0.882, 0.71]`
  - `🏷️ Modality:` `FIGURE (Line Chart)` or `TABLE (Reconstructed 6x5 Matrix)`

### 5.3 Reconstructed Table View
- Styled with alternating row striping (`#0F172A` / `#1E293B`), bold column headers with `#06B6D4` bottom border, and horizontal scroll support for wide multi-column engineering tables.
