nohup python generate_adv_snli_.py --data_path ./data --radius 1.0 --classifier_path ./ --datatype 0 --advmode tail --modelmode cnn --voc_file ./vocab.json >> ./gen_snli_tail_cnn.log 2>&1 &