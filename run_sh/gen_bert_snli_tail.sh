python bert_adv_snli.py \
--data_path ./data \
--radius 1.5 \
--classifier_path ./ \
--datatype 0 \
--advmode tail \
--voc_file ./vocab.json \
>> ./gen_bert_snli_tail_1.5.log 2>&1 &
