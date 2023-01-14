python bert_adv_snli.py \
--data_path ./data \
--radius 4.0 \
--classifier_path ./ \
--datatype 0 \
--advmode all \
--voc_file ./vocab.json \
>> ./bert_snli_all_4.0.log 2>&1 &
