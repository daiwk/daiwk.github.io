#pandoc -N -s --toc --smart --latex-engine=xelatex -V CJKmainfont='微软雅黑' -V mainfont='Times New Roman' -V geometry:margin=1in ./assets/int.md -o ./assets/int.pdf
#pandoc -N -s --toc --smart --latex-engine=xelatex -V CJKmainfont='微软雅黑' -V mainfont='Times New Roman' -V geometry:margin=1in ./assets/int-ml.md -o ./assets/int-ml.pdf


#pandoc -N -s --toc --smart --latex-engine=xelatex -V CJKmainfont='黑体-简' -V mainfont='Times New Roman' -V geometry:margin=1in ./assets/int.md -o ./assets/int.pdf
#pandoc -N -s --toc --smart --latex-engine=xelatex -V CJKmainfont='黑体-简' -V mainfont='Times New Roman' -V geometry:margin=1in ./assets/int-ml.md -o ./assets/int-ml.pdf

pandoc -N -s --toc --smart --latex-engine=xelatex -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in -f markdown+markdown_in_html_blocks+raw_html-implicit_figures ./assets/int.md -o ./assets/int.pdf
pandoc -N -s --toc --smart --latex-engine=xelatex -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in  -f markdown+markdown_in_html_blocks+raw_html-implicit_figures./assets/int-ml.md -o ./assets/int-ml.pdf

cat ./_posts/2019-02-02-dl-graph-representations.md | python3 trans_format.py > ./assets/graph-representations.md
pandoc -N -s --toc --smart --latex-engine=xelatex -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in -f markdown+markdown_in_html_blocks+raw_html-implicit_figures ./assets/graph-representations.md -o ./assets/graph-representations.pdf

# find   

