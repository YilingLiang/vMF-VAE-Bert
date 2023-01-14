python bert_adv_snli.py \
--data_path ./data \
--radius 3.0 \
--classifier_path ./ \
--datatype 0 \
--advmode fore \
--voc_file ./vocab.json \
>> ./gen_bert_snli_fore_3.0.log 2>&1 &
