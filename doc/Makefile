.PHONY:clean all
all:
	xelatex report.tex; bibtex report.aux; xelatex report.tex; xelatex report.tex
clean:
	-rm -rf *.aux *.bbl *.blg *.out *.log *.pdf
