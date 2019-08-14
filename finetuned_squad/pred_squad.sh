SQUAD_DIR=/home/gene/qademo/tmp
#--model_type bert \
#--model_name_or_path bert-base-uncased \
#--train_file $SQUAD_DIR/train-v2.0.json \
#    --version_2_with_negative \
#    --do_eval \
#    --model_type bert \
#    --model_name_or_path ../../clouddrive \
#    --do_lower_case \
#    --predict_file $SQUAD_DIR/eval.json \
#    --max_seq_length 512 \
#    --doc_stride 128 \
#    --output_dir ../../clouddrive \
#    --n_best_size 10 \
#    --no_cuda \

#python -m torch.distributed.launch --nproc_per_node=4 ./examples/run_squad.py \
python ./examples/run_squad.py
