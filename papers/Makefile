# Builds the atlas paper. All headline numbers come from generated macros:
#   python3 scripts/sync_readme.py   (repo root; writes results/derived/macros.tex)
# The paper reads ../results/derived/macros.tex at compile time; a placeholder
# keeps it buildable before experiments have run.
#
# Requires any LaTeX distribution with acmart (TeX Live or tectonic).
# Tectonic one-liner (no root):  mkdir -p "$HOME/.local/bin" && \
#   curl -sL https://github.com/tectonic-typesetting/tectonic/releases/latest/download/tectonic-*-x86_64-unknown-linux-musl.tar.gz | tar xz -C "$HOME/.local/bin"

TECTONIC ?= tectonic
LATEXMK  ?= latexmk

.PHONY: all tectonic latex clean

all: tectonic

tectonic:
	$(TECTONIC) -X compile main.tex

latex:
	$(LATEXMK) -pdf -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.pdf main.out
