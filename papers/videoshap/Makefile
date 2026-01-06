# Makefile for VideoSHAP paper

MAIN = videoshap
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib cvpr.sty
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf || open $(MAIN).pdf

# Quick compile without bibtex (for drafting)
quick:
	$(LATEX) $(MAIN)

# Word count
wc:
	@texcount -inc -total $(MAIN).tex
