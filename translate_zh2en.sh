devices=0
who=test
#who=valid
model_root_dir=output
task=zh2en
model_dir_tag=zh2en_baseline
dir_bleu=$model_root_dir/$task/$model_dir_tag
dir=$model_dir_tag
files=`ls $dir_bleu/checkpoint_best.pt`
#files=`ls $dir_bleu/last5.ensemble.pt`
echo $flies
ckpts=($files)
for ckpt in ${ckpts[@]}
do
  echo -e "\033[34mckpt=${ckpt}\033[0m"
  sh translate_decode.sh ${devices} ${dir} ${who} ${ckpt}
  wait
  perl multi-bleu.perl -lc zh2en_ref/$who/$who* < $dir_bleu/hypo.sorted
done

