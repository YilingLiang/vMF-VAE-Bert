python bert_adv_qqp.py \
--data_path ./data \
--radius 1.5 \
--classifier_path ./ \
--datatype 1 \
--advmode fore \
--voc_file ./vocab.json \
>> ./gen_bert_qqp_fore_1.5.log 2>&1 &
