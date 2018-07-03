tail -n 1 maskbragg*.out | sed 's/=//g' | tr '<\n' ' ' | tr '>' '\n' | sed 's/maskbragg//g' | sed 's/.out//g' | sort -nk1,1 | wc -l
