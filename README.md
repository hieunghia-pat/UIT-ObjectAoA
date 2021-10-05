Object Relation Transformer for Vietnamese Image Captioning
=================
This repo is based on [Object Relation Transformer](https://github.com/yahoo/object_relation_transformer) with modification for using with Python3 and the [UIT-ViIC](https://arxiv.org/abs/2002.00175) dataset. For more detail please visit the original repository.

From there to the end there are instructions of the original repository.

## Object Relation Transformer

This is a PyTorch implementation of the Object Relation Transformer published in NeurIPS 2019. You can find the paper [here](https://papers.nips.cc/paper/9293-image-captioning-transforming-objects-into-words.pdf). This repository is largely based on code from Ruotian Luo's Self-critical Sequence Training for Image Captioning GitHub repo, which can be found [here](https://github.com/ruotianluo/self-critical.pytorch).

The primary additions are as follows:
* Relation transformer model
* Script to create reports for runs on MSCOCO


### Requirements
* Python 3.x
* PyTorch 1.4+ (along with torchvision)
* h5py
* scikit-image
* typing
* pyemd
* gensim
* gdown

For [coco-caption](https://github.com/daqingliu/coco-caption) and [cider](https://github.com/ruotianluo/cider) we added it with modification to work with Python3 and the UIT-ViIC dataset. So you don't need to get it as the requirement in the original repository.

### Data Preparation

#### Download pre-processed caption and extracted feature

First download the requirement for `coco-caption` part
```
cd coco-caption
sh get_stanford_models.sh
```

Download the generated data as the following command line:
```
gdown --id 1lqzulCSU0hWKf6Fp4Au2ltZwo_c0YSR6
```

Then unzip it inside the UIT-ObjectRelationTransformer directory.

```
unzip data.zip
rm data.zip
```

In this folder we have all required features and pre-processed UIT-ViCI caption.

#### Download the UIT-ViIC dataset

```
gdown --id 1p1yAQrvqdssjsLlFyr_K_oOiUSWXIcz-
unzip UIT-ViIC.zip
rm UIT-ViIC.zip
```

### Model Training and Evaluation

#### Standard cross-entropy loss training

```
python train.py --id relation_transformer_bu \
                --caption_model relation_transformer \
                --input_json data/viecap4htalk.json \
                --input_fc_dir data/viecap4hbu_fc \
                --input_att_dir data/viecap4hbu_att \
                --input_box_dir data/viecap4hbu_box \
                --input_rel_box_dir data/viecap4hbu_box_relative \
                --input_label_h5 data/viecap4htalk_label.h5 \
                --checkpoint_path viecap_log_relation_transformer_bu \
                --noamopt --noamopt_warmup 10000 \
                --label_smoothing 0.1 \
                --batch_size 10 \
                --learning_rate 5e-4 \
                --num_layers 6 \
                --input_encoding_size 512 \
                --rnn_size 2048 \
                --learning_rate_decay_start 0 \
                --scheduled_sampling_start 0 \
                --save_checkpoint_every 200 \
                --language_eval 0 \
                --val_images_use -1 \
                --max_epochs 40 \
                --use_box 1
```

The train script will dump checkpoints into the folder specified by `--checkpoint_path` (default = `save/`). We only save the best-performing checkpoint on validation and the latest checkpoint to save disk space.

To resume training, you can specify `--start_from` option to be the path saving `infos.pkl` and `model.pth` (usually you could just set `--start_from` and `--checkpoint_path` to be the same).

Currently we disabled the histories object while training for optimize the usage of memory while when train the model on Google Colab. If you have tensorflow and want to track the training process, enable the histories object in [train.py](train.py) and the loss histories are automatically dumped into `--checkpoint_path`, and can be visualized using tensorboard.

The current command uses scheduled sampling. You can also set scheduled_sampling_start to -1 to disable it.

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to download the [coco-caption code](https://github.com/tylin/coco-caption) into `coco-caption` directory.

For more options, see `opts.py`.

#### Self-critical RL training

After training using cross-entropy loss, additional self-critical training produces signficant gains in CIDEr-D score.

First, copy the model from the pretrained model using cross entropy. (It's not mandatory to copy the model, just for back-up)

```
bash scripts/copy_model.sh relation_transformer_bu relation_transformer_bu_rl
```

Then:

```
python train.py   --id relation_transformer_bu_rl \
                  --caption_model relation_transformer \
                  --input_json data/uitviictalk.json \
                  --input_fc_dir data/uitviicbu_fc \
                  --input_att_dir data/uitviicbu_att \
                  --input_label_h5 data/uitviictalk_label.h5  \
                  --input_box_dir data/uitviicbu_box \
                  --input_rel_box_dir data/uitviicbu_box_relative \
                  --input_label_h5 data/uitviictalk_label.h5 \
                  --checkpoint_path log_relation_transformer_bu_rl \
                  --batch_size 10 \
                  --learning_rate 5e-4 \
                  --num_layers 6 \
                  --input_encoding_size 512 \
                  --rnn_size 2048 \
                  --learning_rate_decay_start 0 \
                  --scheduled_sampling_start 0 \
                  --start_from log_relation_transformer_bu_rl \
                  --save_checkpoint_every 1000 \
                  --language_eval 1 \
                  --val_images_use -1 \
                  --self_critical_after 30 \
                  --max_epochs 60 \
                  --use_box 1 \
                  --legacy_extra_skip 1
```

#### Evaluate on test split
To evaluate the cross-entropy model, run:

```python
python eval.py    --dump_images 0 \
                  --num_images -1 \
                  --model viecap_log_relation_transformer_bu/model-best.pth \
                  --infos_path viecap_log_relation_transformer_bu/infos_relation_transformer_bu-best.pkl \
                  --image_root /home/nguyennghia/Projects/VieCap4H \
                  --input_json data/viecap4htalk.json \
                  --input_label_h5 data/viecap4htalk_label.h5 \
                  --input_fc_dir test_data/viecap4hbu_fc \
                  --input_att_dir test_data/viecap4hbu_att \
                  --input_box_dir test_data/viecap4hbu_box \
                  --input_rel_box_dir test_data/viecap4hbu_box_relative \
                  --language_eval 1
```

and for cross-entropy+RL run:

```python
python eval.py    --dump_images 0 \
                  --num_images -1 \
                  --model log_relation_transformer_bu_rl/model-best.pth \
                  --infos_path log_relation_transformer_bu_rl/infos_relation_transformer_bu_rl-best.pkl \
                  --image_root /content/UIT-ViIC \
                  --input_json data/uitviictalk.json \
                  --input_label_h5 data/uitviictalk_label.h5  \
                  --input_fc_dir data/uitviicbu_fc \
                  --input_att_dir data/uitviicbu_att \
                  --input_box_dir data/uitviicbu_box \
                  --input_rel_box_dir data/uitviicbu_box_relative \
                  --language_eval 1 \
                  --beam_size 5
```

## predict on test split

```python
python predict.py --dump_images 0 \
                  --num_images -1 \
                  --model viecap_log_relation_transformer_bu/model-best.pth \
                  --caption_model relation_transformer \
                  --infos_path viecap_log_relation_transformer_bu/infos_relation_transformer_bu-best.pkl \
                  --input_json_talk data/viecap4htalk.json \
                  --input_json test_data/test.json \
                  --input_label_h5 data/viecap4htalk_label.h5 \
                  --input_fc_dir test_data/viecap4hbu_fc \
                  --input_att_dir test_data/viecap4hbu_att \
                  --input_box_dir test_data/viecap4hbu_box \
                  --input_rel_box_dir test_data/viecap4hbu_box_relative \
                  --language_eval 1
```

### Model Zoo and Results

The table below presents links to our pre-trained models, as well as results from our paper on the Karpathy test
split. Similar results should be obtained by running the respective commands in
[neurips_training_runs.sh](neurips_training_runs.sh). As learning rate scheduling was not fully optimized, these
values should only serve as a reference/expectation rather than what can be achieved with additional tuning.

The models are Copyright Verizon Media, licensed under the terms of the CC-BY-4.0 license. See associated
[license file](LICENSE-CC-BY-4.md).

Algorithm | CIDEr-D |SPICE | BLEU-1 | BLEU-4 | METEOR | ROUGE-L
:-- | :--: | :--: | :--: | :--: | :--: | :--:
[Up-Down + LSTM](http://wpc.D89ED.chicdn.net/00D89ED/object_relation_transformer/models/log_topdown_bu.zip) * | 106.6 | 19.9 | 75.6 | 32.9 | 26.5 | 55.4
[Up-Down + Transformer](http://wpc.D89ED.chicdn.net/00D89ED/object_relation_transformer/models/log_transformer_bu.zip) | 111.0 | 20.9 | 75.0 | 32.8 | 27.5 | 55.6
[Up-Down + Object Relation Transformer](http://wpc.D89ED.chicdn.net/00D89ED/object_relation_transformer/models/log_relation_transformer_bu.zip) | 112.6 | 20.8 | 75.6 |33.5 |27.6 | 56.0
[Up-Down + Object Relation Transformer](http://wpc.D89ED.chicdn.net/00D89ED/object_relation_transformer/models/log_relation_transformer_bu.zip) + Beamsize 2 | 115.4 | 21.2 | 76.6 | 35.5 | 28.0 | 56.6
[Up-Down + Object Relation Transformer + Self-Critical](http://wpc.D89ED.chicdn.net/00D89ED/object_relation_transformer/models/log_relation_transformer_bu_rl.zip) + Beamsize 5 | 128.3 | 22.6 | 80.5 | 38.6 | 28.7 | 58.4

\* Note that the pre-trained Up-Down + LSTM model above produces slightly better results than
reported, as it came from a different training run. We kept the older LSTM results in the table above for consistency
with our paper.

#### Comparative Analysis

In addition, in the paper we also present a head-to-head comparison of the Object Relation Transformer against the "Up-Down + Transformer" model. (Results from the latter model are also included in the table above).
In the paper, we refer to this latter model as "Baseline Transformer", as it does not make use of geometry in its attention definition. The idea of the head-to-head comparison is to better understand the improvement
obtained by adding geometric attention to the Transformer, both quantitatively and qualitatively. The comparison consists of a set of evaluation metrics computed for each model on a per-image basis, as well as aggregated over all images.
It includes the results of paired t-tests, which test for statistically significant differences between the evaluation metrics resulting from each of the models. This comparison can be generated by running the commands in
[neurips_report_comands.sh](neurips_report_commands.sh). The commands first run the two aforementioned models on the MSCOCO test set and then generate the corresponding report containing the complete comparative analysis.


### Citation

If you find this repo useful, please consider citing (no obligation at all):

```
@article{herdade2019image,
  title={Image Captioning: Transforming Objects into Words},
  author={Herdade, Simao and Kappeler, Armin and Boakye, Kofi and Soares, Joao},
  journal={arXiv preprint arXiv:1906.05963},
  year={2019}
}
```

Of course, please cite the original paper of models you are using (you can find references in the model files).

### Contribute

Please refer to [the contributing.md file](Contributing.md) for information about how to get involved. We welcome
issues, questions, and pull requests.

Please be aware that we (the maintainers) are currently busy with other projects, so it make take some days before we
are able to get back to you. We do not foresee big changes to this repository going forward.

### Maintainers

Kofi Boakye: kaboakye@verizonmedia.com

Simao Herdade: sherdade@verizonmedia.com

Joao Soares: jvbsoares@verizonmedia.com

### License

This project is licensed under the terms of the MIT open source license. Please refer to [LICENSE](LICENSE) for the full terms.


### Acknowledgments

Thanks to [Ruotian Luo](https://github.com/ruotianluo) for the original code.
