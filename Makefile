.PHONY:lib lib-clean doc doc-clean
lib:
	make -C steadyNS
lib-clean:
	make -C steadyNS clean
doc:
	make -C doc
doc-clean:
	make -C doc clean
