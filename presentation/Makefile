# Adapted from https://github.com/agoldst/tex

# output filename will be name-suffix.pdf
nup_suffix := 4up
nup_layout := 2x2
# for portrait, set to --no-landscape
nup_landscape := --landscape

notes.pdf: slides.md preamble-notes.tex
	pandoc $< \
        --data-dir . \
	    -t beamer \
	    --slide-level=2 \
	    -H preamble-notes.tex \
	    --latex-engine xelatex \
	    -o $@

slides.tex: slides.md common.tex
	pandoc $< \
	    -t beamer \
	    -H common.tex \
        --data-dir . \
	    -V theme:m \
	    -V themeopts:usetitleprogressbar,nooffset \
	    --slide-level=2 \
	    --latex-engine xelatex \
	    -o $@

slides.pdf: slides.md common.tex
	pandoc $< \
	    -t beamer \
        --data-dir . \
	    -V theme:m \
	    -V themeopts:usetitleprogressbar,nooffset \
	    --slide-level=2 \
	    -H common.tex \
	    --latex-engine xelatex \
	    -o $@

pdfs := notes.pdf slides.pdf
nups := $(patsubst %.pdf,%-$(nup_suffix).pdf,$(pdfs))

$(nups): %-$(nup_suffix).pdf: %.pdf
	pdfjam $< \
	    $(nup_landscape) \
	    --nup $(nup_layout) \
	    --suffix $(nup_suffix)

all: $(pdfs) $(nups)

clean:
	rm -f $(pdfs) $(nups)

.DEFAULT_GOAL := all

.PHONY: all clean

