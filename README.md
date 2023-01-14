### vMF-VAE-Bert

Main dependency:

```
bert4keras
numpy=1.18.5
tensorflow=1.15.5
pytorch=1.7.1
transformers=4.9.1
nltk=3.6.6
```

To get the generative model, in `gen_model`:

```shell
python main_qqp_snli.py
```

Generate adversarial examples: 

```shell
python bert_adv_qqp.py \
  --data_path ./data \
  --radius 1.5 \
  --classifier_path ./ \
  --datatype 1 \
  --advmode fore \
  --voc_file ./vocab.json \
  >> ./gen_bert_qqp_fore_1.5.log 2>&1 &
```

More scripts are in folder `run_sh` . 