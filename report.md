# Project Report & Slide Plan

Deadline: 5/25 | 5 min talk + 2 min Q&A | 5-7 slides

---

## Slide Content

### Slide 1 — Project Description
**Title**: Text-Guided 3D Object Stylization via VLM Image Editing

One sentence: *Given a single photo, generate multiple 3D objects with different appearances by combining instruction-based 2D editing with single-image 3D reconstruction.*

Put the teaser figure here (see Demo Plan below).

---

### Slide 2 — Motivation
- 3D content creation is expensive and requires expertise
- Text-to-3D models are slow or low quality
- **But**: 2D image editing (VLMs) is already very strong
- **And**: Single-view image-to-3D (DiffusionGS) is fast and decent quality
- **Gap**: No simple way to get the same object in multiple styles as 3D

> Insight: edit in 2D where models are strongest, then lift to 3D

---

### Slide 3 — Method (Pipeline)

```
Input Photo
     ↓
[FLUX Kontext]  instruction prompt
     ↓  (edit in 2D, strongest signal)
Edited Image
     ↓
[DiffusionGS]  image → 3D Gaussian Splatting
     ↓
3D Object  →  turntable video / PLY / OBJ
```

Key design decision: **sequential staging** — VLM loads, edits, unloads → then DiffusionGS runs. Avoids VRAM conflict on a single consumer GPU.

---

### Slide 4 — Results: Style Grid

One object × 4 styles.

Put a 2×2 grid of turntable frame strips:
- Original 3D (no edit)
- Gold / metallic
- Marble / stone
- Clay / matte

Each cell: 4-frame turntable strip (0°, 90°, 180°, 270°)

---

### Slide 5 — Results: Ablation

**Why VLM stage matters** — same object, same 3D model, different input:

| | Input to DiffusionGS | 3D result |
|---|---|---|
| Baseline | original photo | original appearance |
| Ours | VLM-edited photo | styled appearance |

Show side-by-side: original image → original 3D vs. edited image → styled 3D.
This directly proves the VLM stage contributes.

---

### Slide 6 — Experience & Thoughts

**What worked**
- VLM 2D editing (FLUX Kontext) handles complex instructions well
- DiffusionGS faithfully reconstructs the edited appearance into 3D
- Sequential VRAM staging makes it run on a single GPU

**Limitations**
- Geometry is fixed from the original image — VLM can change appearance but 3D shape follows the original photo
- 3D quality depends on DiffusionGS, which struggles with thin structures or transparent objects

**Future directions**
- SDS-based geometry editing after reconstruction
- Text-only input: LLM auto-generates style variants

---

## Demo Plan

4 objects × 3 prompts each. Each prompt generates: `vlm_edited.png` (2D edit) + `output.gif` (3D turntable).

---

### Object 1 — popmart.png (chibi blue hoodie figure)
```bash
# Baseline (no edit) — for ablation slide
python edit_3d.py --image popmart.png --output_dir out/popmart_baseline --no_mesh

# Material 1: gold
python edit_3d.py --image popmart.png \
  --vlm_prompt "turn the hoodie and pants into polished gold metal, keep the face skin-toned" \
  --output_dir out/popmart_gold --no_mesh

# Material 2: marble
python edit_3d.py --image popmart.png \
  --vlm_prompt "turn the entire figure into white marble with grey veins, remove all clothing details" \
  --output_dir out/popmart_marble --no_mesh

# Creative: obsidian with lava cracks
python edit_3d.py --image popmart.png \
  --vlm_prompt "make the surface black obsidian stone with glowing orange lava cracks running across the body" \
  --output_dir out/popmart_lava --no_mesh
```

---

### Object 2 — avocado.png (half avocado)
```bash
# Baseline
python edit_3d.py --image avocado.png --output_dir out/avocado_baseline --no_mesh

# Material 1: gold
python edit_3d.py --image avocado.png \
  --vlm_prompt "make the avocado shell brushed gold metal, keep the pit dark brown" \
  --output_dir out/avocado_gold --no_mesh

# Material 2: crystal
python edit_3d.py --image avocado.png \
  --vlm_prompt "make the avocado flesh transparent green crystal, keep the pit as dark amber gemstone" \
  --output_dir out/avocado_crystal --no_mesh

# Creative: jade artifact
python edit_3d.py --image avocado.png \
  --vlm_prompt "turn the avocado into carved jade, add small dragon patterns on the shell surface" \
  --output_dir out/avocado_jade --no_mesh
```

---

### Object 3 — monster.png (blue fluffy monster toy)
```bash
# Baseline
python edit_3d.py --image monster.png --output_dir out/monster_baseline --no_mesh

# Material 1: porcelain
python edit_3d.py --image monster.png \
  --vlm_prompt "make the monster a white porcelain figurine with hand-painted blue floral patterns on the body" \
  --output_dir out/monster_porcelain --no_mesh

# Material 2: wood
python edit_3d.py --image monster.png \
  --vlm_prompt "make the monster carved from rough wood, replace the fur texture with visible wood grain" \
  --output_dir out/monster_wood --no_mesh

# Creative: bioluminescent creature
python edit_3d.py --image monster.png \
  --vlm_prompt "make the monster glow purple and blue from within like a deep sea bioluminescent creature, add glowing spots on the fur" \
  --output_dir out/monster_biolum --no_mesh
```

---

### Object 4 — doctor.png (chibi Doctor Strange figure)
```bash
# Baseline
python edit_3d.py --image doctor.png --output_dir out/doctor_baseline --no_mesh

# Material 1: bronze with patina
python edit_3d.py --image doctor.png \
  --vlm_prompt "turn the cloak and armor into ancient bronze, add green patina in the robe folds and recesses" \
  --output_dir out/doctor_bronze --no_mesh

# Material 2: clay toy
python edit_3d.py --image doctor.png \
  --vlm_prompt "make it a handmade clay toy, replace the cloak with hand-painted earthy red and ochre colors" \
  --output_dir out/doctor_clay --no_mesh

# Creative: terracotta warrior
python edit_3d.py --image doctor.png \
  --vlm_prompt "turn it into a Chinese terracotta warrior, replace the cloak with armored plates, add faded red paint on the face" \
  --output_dir out/doctor_terracotta --no_mesh
```

---

## What to Capture for Slides

- `vlm_edited.png` — 2D edit result (show in ablation + pipeline diagram)
- `output.gif` — 3D turntable (embed if PowerPoint supports GIF; else screenshot at 0°/90°/180°/270°)

**Slide 1 teaser**: pick popmart or monster, show 1 row:
```
original photo → FLUX edit (gold) → 3D frames
```

**Slide 4 style grid**: one object (popmart recommended), 2×2 grid:
```
baseline | gold
marble   | lava
```
Each cell = 4-frame turntable strip.

**Slide 5 ablation**: popmart baseline (original photo → plain 3D) vs. popmart gold (gold photo → gold 3D), side by side.

---

## Run Order (efficiency)

Run baselines first — they're fastest (no VLM call).
Then creative prompts — those take longest on Replicate.

```bash
python edit_3d.py --image popmart.png --output_dir out/popmart_baseline --no_mesh
python edit_3d.py --image avocado.png --output_dir out/avocado_baseline --no_mesh
python edit_3d.py --image monster.png --output_dir out/monster_baseline --no_mesh
python edit_3d.py --image doctor.png  --output_dir out/doctor_baseline  --no_mesh
```

---

## FLUX Kontext Prompt Tips

**基本格式**: `[指定部位] + [材質/顏色變化] + [可選：新增或移除的元素]`

### 原則：指定部位，描述變化

不要描述整體材質，要指出**哪個部分**變成什麼，以及**保持**哪個部分不動：

| 不好（整體描述） | 好（指定部位） |
|--------------|-------------|
| `make it solid gold` | `turn the hoodie into polished gold, keep the face skin-toned` |
| `white marble sculpture` | `turn the entire figure into white marble, remove all clothing details` |
| `make it glow` | `add glowing blue spots on the fur, make the skin emit purple light` |
| `bronze statue` | `turn the cloak into ancient bronze, add green patina in the folds` |

新增或移除元素也可以直接說：
- `add dragon patterns on the shell`
- `replace the fur texture with wood grain`
- `remove the cape, add armor plates instead`