<p align="center">
  <img src="images/genai_shap.png" alt="GenAI-SHAP: Unified Explainability Framework" width="700"/>
</p>

<h1 align="center">GenAI-SHAP</h1>
<h3 align="center">Unified Explainability for Generative AI</h3>

<p align="center">
A framework for interpreting modern AI systems using Monte Carlo Shapley value estimation.<br/>
Model-agnostic explainability across language models, vision-language models, video understanding, and autonomous agents.
</p>

---

## Overview

GenAI-SHAP provides a unified theoretical foundation for explaining generative AI systems. All methods share the same core principle: **Shapley values from cooperative game theory**. By treating input components (tokens, objects, tools, tracked entities) as players in a cooperative game, we rigorously quantify each component's contribution to the model's output.

| Method | Domain | Input Components | Publication |
|--------|--------|------------------|-------------|
| [**TokenSHAP**](#tokenshap) | Large Language Models | Text tokens | [arXiv:2407.10114](https://arxiv.org/abs/2407.10114) |
| [**PixelSHAP**](#pixelshap) | Vision-Language Models | Visual objects | [arXiv:2503.06670](https://arxiv.org/abs/2503.06670) |
| [**VideoSHAP**](#videoshap) | Video Understanding | Tracked objects | Coming soon |
| [**AgentSHAP**](#agentshap) | LLM Agents | Tools | [arXiv:2512.12597](https://arxiv.org/abs/2512.12597) |

### Why Shapley Values?

Shapley values provide the **unique** attribution method satisfying four axiomatic properties:

- **Efficiency** — Attributions sum to the total value
- **Symmetry** — Equal contributors receive equal attribution
- **Null player** — Non-contributors receive zero attribution
- **Linearity** — Attributions combine predictably across games

These guarantees make Shapley-based explanations theoretically grounded and practically meaningful.

---

<h2 align="center">
  <img src="images/tokenshap_logo.png" alt="TokenSHAP Logo" width="80" style="vertical-align: middle;"/>
  <br/>TokenSHAP
</h2>

<p align="center"><em>Interpreting Large Language Models through Token Attribution</em></p>

TokenSHAP quantifies how individual tokens in a prompt influence an LLM's response by systematically ablating tokens and measuring output changes.

### Features

- Monte Carlo estimation for computational efficiency
- Support for local models (HuggingFace) and API-based models (OpenAI, Ollama)
- Visual highlighting of token importance
- Configurable sampling strategies
- Multiple text splitting methods (sentence-based, tokenizer-based)

### Quick Start

```python
from token_shap import TokenSHAP, LocalModel, StringSplitter

model = LocalModel("meta-llama/Llama-3.2-3B-Instruct")
token_shap = TokenSHAP(model, StringSplitter())

results = token_shap.analyze(
    prompt="Explain why the sky appears blue during the day.",
    sampling_ratio=0.3,
    print_highlight_text=True
)
```

Output shows token importance with color highlighting: **red** = high importance, **blue** = low importance.

See [`notebooks/TokenShap Examples.ipynb`](notebooks/TokenShap%20Examples.ipynb) for complete examples.

---

<h2 align="center">
  <img src="images/pixelshap_logo.png" alt="PixelSHAP Logo" width="80" style="vertical-align: middle;"/>
  <br/>PixelSHAP
</h2>

<p align="center"><em>Object-Level Attribution for Vision-Language Models</em></p>

PixelSHAP extends Shapley analysis to visual content by treating segmented objects as attribution units. This enables fine-grained understanding of which visual elements drive a VLM's textual response.

### Features

- Model-agnostic: works with any black-box VLM (GPT-4o, Gemini, LLaVA, etc.)
- State-of-the-art segmentation with YOLO + SAM2
- Multiple masking strategies: blackout, blur, inpainting
- Rich visualization: heatmaps, importance rankings, side-by-side comparisons

### Quick Start

```python
from token_shap import PixelSHAP, OpenAIModel, OpenAIEmbeddings
from token_shap import YoloSam3SegmentationModel, BlackoutSegmentationManipulator

# Initialize components
vlm = OpenAIModel(model_name="gpt-4o", api_key="...")
segmentation = YoloSam3SegmentationModel()
manipulator = BlackoutSegmentationManipulator(mask_type='bbox')
embeddings = OpenAIEmbeddings(api_key="...")

# Create analyzer
pixel_shap = PixelSHAP(
    model=vlm,
    segmentation_model=segmentation,
    manipulator=manipulator,
    vectorizer=embeddings
)

# Analyze image
results_df, shapley_values = pixel_shap.analyze(
    image_path="scene.jpg",
    prompt="What is causing the traffic delay?",
    sampling_ratio=0.3
)

# Visualize
pixel_shap.visualize(show_original_side_by_side=True)
```

See [`notebooks/PixelSHAP Examples.ipynb`](notebooks/PixelSHAP%20Examples.ipynb) for complete examples.

---

<h2 align="center">
  <img src="images/videoshap_logo.png" alt="VideoSHAP Logo" width="80" style="vertical-align: middle;"/>
  <br/>VideoSHAP
</h2>

<p align="center"><em>Temporal Object Attribution for Video Understanding</em></p>

VideoSHAP extends the framework to video content, tracking objects across frames and computing importance scores that reflect each object's contribution to the VLM's temporal reasoning.

### Features

- SAM3-based object tracking with text-prompted detection
- Support for native video models (Gemini) and frame-sequence approaches (GPT-4o)
- Temporal importance profiles showing how relevance evolves
- Video output with heatmap overlays
- Adaptive frame sampling with target FPS control

### Quick Start

```python
from token_shap import (
    VideoSHAP,
    SAM3VideoSegmentationModel,
    VideoBlurManipulator,
    GeminiVideoModel
)
from token_shap.base import HuggingFaceEmbeddings

# Initialize tracking and VLM
tracker = SAM3VideoSegmentationModel(model_name="facebook/sam3", device="cuda")
vlm = GeminiVideoModel(model_name="gemini-2.0-flash", api_key="...")
manipulator = VideoBlurManipulator(blur_radius=51)
embeddings = HuggingFaceEmbeddings()

# Create analyzer
video_shap = VideoSHAP(
    model=vlm,
    segmentation_model=tracker,
    manipulator=manipulator,
    vectorizer=embeddings
)

# Analyze video
results_df, shapley_values = video_shap.analyze(
    video_path="video.mp4",
    prompt="What is happening in this scene?",
    text_prompts=["person", "car", "object"],
    target_fps=8,
    sampling_ratio=0.3
)

# Generate visualization
video_shap.visualize(output_path="attributed_video.mp4")
```

### Example Results

<table>
<tr>
<td align="center"><b>Birthday Party Analysis</b></td>
<td align="center"><b>Cats Interaction</b></td>
</tr>
<tr>
<td><img src="images/birthday.gif" alt="Birthday party video analysis" width="400"/></td>
<td><img src="images/cats.gif" alt="Cats video analysis" width="400"/></td>
</tr>
</table>

<table>
<tr>
<td align="center"><b>Scene Understanding Q1</b></td>
<td align="center"><b>Scene Understanding Q2</b></td>
<td align="center"><b>Scene Understanding Q3</b></td>
</tr>
<tr>
<td><img src="images/alien_q1.gif" alt="Alien scene question 1" width="280"/></td>
<td><img src="images/alien_q2.gif" alt="Alien scene question 2" width="280"/></td>
<td><img src="images/alien_q3.gif" alt="Alien scene question 3" width="280"/></td>
</tr>
</table>

See [`notebooks/VideoSHAP Examples.ipynb`](notebooks/VideoSHAP%20Examples.ipynb) for complete examples.

---

<h2 align="center">
  <img src="images/agentshap_logo.png" alt="AgentSHAP Logo" width="80" style="vertical-align: middle;"/>
  <br/>AgentSHAP
</h2>

<p align="center"><em>Tool Attribution for LLM Agents</em></p>

AgentSHAP explains autonomous agent behavior by quantifying each tool's contribution to the agent's response. This enables debugging, optimization, and trust calibration for agentic systems.

<p align="center">
  <img src="images/agent_shap_method.jpeg" alt="AgentSHAP Method" width="700"/>
</p>

### Features

- Measures tool importance through systematic ablation
- Compatible with any function-calling LLM (OpenAI, Anthropic, etc.)
- Identifies redundant tools and critical dependencies
- Supports complex multi-tool workflows

### Quick Start

```python
from token_shap import AgentSHAP, OpenAIModel, OpenAIEmbeddings, create_function_tool

model = OpenAIModel(model_name="gpt-4o-mini", api_key="...")
embeddings = OpenAIEmbeddings(api_key="...")

# Define tools
weather_tool = create_function_tool(
    name="get_weather",
    description="Get current weather for a location",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    executor=lambda args: f"Weather in {args['city']}: 72°F, sunny"
)

calendar_tool = create_function_tool(
    name="check_calendar",
    description="Check calendar events",
    parameters={"type": "object", "properties": {"date": {"type": "string"}}},
    executor=lambda args: f"Events on {args['date']}: Team meeting at 2pm"
)

# Analyze
agent_shap = AgentSHAP(model=model, tools=[weather_tool, calendar_tool], vectorizer=embeddings)
results_df, shapley_values = agent_shap.analyze(
    prompt="Should I bring an umbrella to my meeting tomorrow?",
    sampling_ratio=0.5
)

agent_shap.plot_tool_importance()
```

<p align="center">
  <img src="images/agent_shap_results.jpeg" alt="AgentSHAP Results" width="600"/>
</p>

See [`notebooks/AgentSHAP Examples.ipynb`](notebooks/AgentSHAP%20Examples.ipynb) for complete examples.

---

## Installation

```bash
git clone https://github.com/ronigold/TokenSHAP.git
cd TokenSHAP
pip install -r requirements.txt
```

### Optional Dependencies

```bash
# For local LLMs
pip install transformers accelerate

# For video processing
pip install imageio[ffmpeg]

# For SAM3 tracking
pip install sam3
```

---

## Repository Structure

```
TokenSHAP/
├── token_shap/
│   ├── token_shap.py      # LLM token attribution
│   ├── pixel_shap.py      # VLM object attribution
│   ├── video_shap.py      # Video object attribution
│   ├── agent_shap.py      # Agent tool attribution
│   ├── base.py            # Model abstractions & core algorithms
│   ├── image_utils.py     # Segmentation utilities
│   ├── video_utils.py     # Video processing utilities
│   └── visualization.py   # Plotting utilities
├── notebooks/             # Example Jupyter notebooks
├── experiments/           # Research experiments
├── tests/                 # Test suite
└── images/                # Logos and documentation assets
```

---

## Citation

If you use this framework in your research, please cite the relevant papers:

**TokenSHAP**
```bibtex
@article{goldshmidt2024tokenshap,
  title={TokenSHAP: Interpreting Large Language Models with Monte Carlo Shapley Value Estimation},
  author={Goldshmidt, Roni and Horovicz, Miriam},
  journal={arXiv preprint arXiv:2407.10114},
  year={2024}
}
```

**PixelSHAP**
```bibtex
@article{goldshmidt2025pixelshap,
  title={PixelSHAP: Attention, Please! What Vision-Language Models Actually Focus On},
  author={Goldshmidt, Roni},
  journal={arXiv preprint arXiv:2503.06670},
  year={2025}
}
```

**VideoSHAP**
```bibtex
@article{goldshmidt2025videoshap,
  title={VideoSHAP: Temporal Object Attribution for Video Understanding},
  author={Goldshmidt, Roni},
  journal={arXiv preprint},
  year={2025}
}
```

**AgentSHAP**
```bibtex
@article{horovicz2025agentshap,
  title={AgentSHAP: SHAP-Based Explanation for LLM Agents},
  author={Horovicz, Miriam and Goldshmidt, Roni},
  journal={arXiv preprint arXiv:2512.12597},
  year={2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributors

**Roni Goldshmidt** — roni.goldshmidt@getnexar.com
**Miriam Horovicz** — Co-author of TokenSHAP and AgentSHAP

For issues and feature requests, please use [GitHub Issues](https://github.com/ronigold/TokenSHAP/issues).
