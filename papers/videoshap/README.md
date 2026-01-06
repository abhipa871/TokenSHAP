# VideoSHAP: Temporal Object Attribution for Video Understanding in Vision-Language Models

## Paper Overview

This directory contains the complete LaTeX source for the VideoSHAP paper, a CVPR-style submission on interpreting video-understanding Vision-Language Models through temporal object attribution.

**Author**: Roni Goldshmidt (Nexar)

## Directory Structure

```
videoshap/
├── videoshap.tex      # Main paper (42KB, ~8 pages)
├── references.bib     # 25 BibTeX references
├── cvpr.sty          # CVPR formatting style
├── Makefile          # Build automation
├── README.md         # This file
└── figures/
    ├── teaser.pdf           # Figure 1: Teaser
    ├── pipeline.pdf         # Figure 2: Full pipeline
    ├── ablation_budget.pdf  # Figure 5: Ablation study
    ├── qual_driving.pdf     # Figure 6: Driving example
    ├── qual_sports.pdf      # Figure 7: Sports example
    ├── context_dependent.pdf # Figure 8: Context study
    ├── *.tex                # TikZ source for figures
    └── FIGURES_TODO.md      # Figure specifications
```

## Paper Contents

### Abstract
VideoSHAP extends Shapley-based attribution to video-understanding VLMs by tracking objects across frames and quantifying their influence on model responses.

### Key Contributions
1. **Temporal Object Attribution**: First model-agnostic framework for video VLM interpretation
2. **Multi-Strategy Manipulation**: Blur, blackout, and inpainting approaches evaluated
3. **Comprehensive Evaluation**: Tested on Gemini 2.0, GPT-4o, and Qwen-VL
4. **Open-Source Release**: Complete implementation available

### Main Results
- 64.7% Recall@1 with Gemini 2.0 (blur manipulation)
- 27.9 percentage point improvement over strongest baseline
- Blur manipulation consistently outperforms blackout and inpainting

### Sections
1. Introduction
2. Related Work
3. Problem Statement
4. Methodology (Tracking, Manipulation, Sampling, Shapley)
5. Implementation Details
6. Experimental Setup
7. Results
8. Qualitative Analysis
9. Discussion
10. Conclusion

## Building the Paper

### Prerequisites
- TeX Live or similar LaTeX distribution
- pdflatex, bibtex

### Quick Build
```bash
cd /home/ubuntu/TokenSHAP/papers/videoshap
make
```

### Manual Build
```bash
pdflatex videoshap
bibtex videoshap
pdflatex videoshap
pdflatex videoshap
```

### Clean
```bash
make clean      # Remove auxiliary files
make distclean  # Also remove PDF
```

## Figures

### Placeholder Figures
The current PDFs are matplotlib-generated placeholders. For publication-quality figures:

1. Install a TeX distribution with TikZ/PGFPlots
2. Compile individual figures:
   ```bash
   cd figures
   pdflatex teaser.tex
   pdflatex pipeline.tex
   # etc.
   ```

3. Or replace with actual experimental visualizations from VideoSHAP runs

### Figure Specifications
See `figures/FIGURES_TODO.md` for detailed specifications of each figure.

## Submission Checklist

- [x] Main paper LaTeX source
- [x] BibTeX references (25 citations)
- [x] CVPR style formatting
- [x] Placeholder figures (6 figures)
- [ ] High-resolution final figures
- [ ] Supplementary material
- [ ] Anonymous version (remove author info for blind review)

## Related Files

- **VideoSHAP Implementation**: `/home/ubuntu/TokenSHAP/token_shap/video_shap.py`
- **Example Notebook**: `/home/ubuntu/TokenSHAP/notebooks/VideoSHAP Examples.ipynb`
- **PixelSHAP Paper**: Reference for image-based predecessor

## Citation

```bibtex
@article{goldshmidt2025videoshap,
  title={VideoSHAP: Temporal Object Attribution for Video Understanding
         in Vision-Language Models},
  author={Goldshmidt, Roni},
  journal={arXiv preprint},
  year={2025}
}
```

## Contact

Roni Goldshmidt: roni.goldshmidt@getnexar.com
