ObjectAoA for Vietnamese Image Captioning
=================
This repo is based on [Object Relation Transformer](https://github.com/yahoo/object_relation_transformer) and [Attention on Attention](https://github.com/husthuaan/AoANet) with modification for using with Python3 and the vieCap4H dataset. For more detail please visit the original repositories.

### Requirements
* Python 3.x
* PyTorch 1.4+ (along with torchvision)
* h5py
* scikit-image
* typing
* pyemd
* gensim
* gdown

#### Metrics

We use [coco-caption](https://github.com/daqingliu/coco-caption) and [cider](https://github.com/ruotianluo/cider) for calculating BLEU score, but there exists some modifications you have to make for training and evaluating the ObjectAoA.

#### Get the vieCap4H dataset

```
gdown --id 12FS7g2hFeHVjgrmsalNaaMOGcJwkn8aj
unzip VieCap4H.zip
rm VieCap4H.zip
```

#### Get the pre-processed captions and extracted features

```
gdown --id 1nI6qOxADKwBhxxJY7JQo2i-QW9hqO8ye
unzip data.zip
rm data.zip
```

In this `data` folder we have all json files and feature files for training ObjectAoA on the vieCap4H dataset.


### Model Training and Evaluation

#### Standard cross-entropy loss training

```
python3 train.py  --id relation_transformer_bu_rl \
                  --caption_model relation_transformer \
                  --input_json data/viecap4htalk.json \
                  --input_fc_dir data/viecap4hbu_fc \
                  --input_att_dir data/viecap4hbu_att \
                  --input_label_h5 data/viecap4htalk_label.h5  \
                  --input_box_dir data/viecap4hbu_box \
                  --input_rel_box_dir data/viecap4hbu_box_relative \
                  --checkpoint_path <path-to-your-checkpoint-folder> \
                  --batch_size 64 \
                  --learning_rate 5e-4 \
                  --num_layers 6 \
                  --input_encoding_size 512 \
                  --rnn_size 2048 \
                  --learning_rate_decay_start 0 \
                  --scheduled_sampling_start 0 \
                  --save_checkpoint_every 100 \
                  --language_eval 1 \
                  --val_images_use -1 \
                  --self_critical_after 5 \
                  --max_epochs 30 \
                  --use_box 1 \
                  --legacy_extra_skip 1
```

#### Evaluate in the validation split
```
python3 eval.py  --dump_images 0 \
                  --num_images -1 \
                  --model <path-to-your-checkpoint-folder>/model-best.pth \
                  --infos_path <path-to-your-checkpoint-folder>/infos_relation_transformer_bu_rl-best.pkl \
                  --image_root <image-folder> \
                  --input_json data/viecap4htalk.json \
                  --input_fc_dir data/viecap4hbu_fc \
                  --input_att_dir data/viecap4hbu_att \
                  --input_label_h5 data/viecap4htalk_label.h5  \
                  --input_box_dir data/viecap4hbu_box \
                  --input_rel_box_dir data/viecap4hbu_box_relative \
                  --language_eval 1 \
                  --beam_size 2
```
