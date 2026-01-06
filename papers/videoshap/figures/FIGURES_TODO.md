# Figures Required for VideoSHAP Paper

## 1. teaser.pdf (Figure 1)
**Size**: Single column width (~3.25 inches)
**Content**:
- Left panel: 3-4 key frames from dashcam collision video
- VLM response text: "The Amazon delivery van approaching from the right..."
- Right panel: Same frames with heatmap overlay
- Color scale: Red (high importance) → Green (low importance)
- Amazon van highlighted in red, other vehicles in green/yellow

## 2. pipeline.pdf (Figure 2)
**Size**: Double column width (~7 inches)
**Content**:
Flow diagram showing:
1. Input: Video frames + Text prompts ("vehicle", "person")
2. Stage 1: SAM3 tracking → Tracked objects with IDs
3. Stage 2: Video manipulation → Masked variants
4. Stage 3: VLM inference → Response collection
5. Stage 4: Shapley computation → Importance scores
6. Output: Heatmap visualization + ranked objects

## 3. ablation_budget.pdf (Figure 5)
**Size**: Single column width
**Content**:
- Line plot
- X-axis: "Number of Coalitions" (10, 20, 30, 40, 50, 75, 100)
- Y-axis: "Recall@1 (%)" (40-70 range)
- Solid line: VideoSHAP performance
- Dashed horizontal line: Essential coalitions only (~58%)
- Shaded region showing diminishing returns

## 4. qual_driving.pdf (Figure 6)
**Size**: Single column width
**Content**:
- Top: 4 key frames from driving video
- Middle: Same frames with importance heatmap
- Bottom: Bar chart of object importance scores
  - Amazon van: 0.42 (red)
  - Ego vehicle: 0.18 (orange)
  - Other vehicles: <0.10 (green)
- Query text shown: "Which vehicle will cause collision?"

## 5. qual_sports.pdf (Figure 7)
**Size**: Single column width
**Content**:
- Top: Soccer video frames showing goal sequence
- Middle: Frames with player tracking and importance
- Bottom: Temporal profile showing importance spike at goal frame
- Query: "Who scored the goal?"
- Scorer highlighted in red

## 6. context_dependent.pdf (Figure 8)
**Size**: Single column width
**Content**:
- Same parking lot video, 3 different analyses:
- Row 1: "Which car is backing out?" → White SUV highlighted
- Row 2: "Who is walking to store?" → Pedestrian highlighted
- Row 3: "What vehicle just arrived?" → Blue sedan highlighted
- Show how same video yields different attributions

## Style Notes
- Use consistent color scheme across all figures
- Importance colormap: RdYlGn_r (Red=high, Yellow=mid, Green=low)
- Font: Sans-serif, consistent with paper
- Resolution: 300+ DPI for raster elements
- Vector graphics preferred where possible
