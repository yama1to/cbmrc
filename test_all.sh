#!/bin/sh
DIR="$HOME/cbm_rc2"
Tmp=$DIR/tmp.log 

rm $Tmp
# ls test*cbmrc*.py >> $Tmp
# ls test*esn*.py >> $Tmp
ls test*.py >> $Tmp


while read fname
do
    python3 $fname

done <${Tmp}
cat $Tmp
rm $Tmp
