
# Editing Classifiers + Transformers

This repo extends the work:
**Editing a classifier by rewriting its prediction rules** <br>
*Shibani Santurkar\*, Dimitris Tsipras\*, Mahi Elango, David Bau, Antonio Torralba, Aleksander Madry* <br>
Paper: https://arxiv.org/abs/2112.01008 <br>

The idea is to extend this technique to transformers as well as try applying this to a few other scenarios

### Training
To train a model on one of the datasets normally, just run:
`python train.py --lr 0.01 --checkpoint-name [CHECKPOINT_NAME] --dataset [DATASET] --model [MODEL]`
For example:
`python train.py --lr 0.01 --checkpoint-name Waterbirds-vit --dataset Waterbirds --model ViT`

### Editing CNNs
To edit a classifier using the original papers method:
`python edit.py --layernum 12 --dataset ImageNet --arch vgg16 --rewrite-mode editing`

You can also run their baselines via:
`python edit.py --layernum 12 --dataset ImageNet --arch vgg16 --rewrite-mode finetune_local`
`python edit.py --layernum 12 --dataset ImageNet --arch vgg16 --rewrite-mode finetune_global`

I also added in one more baseline `layer_loss`, which is simply an L2 loss between the layers of the network (so their method but simplified and does not use an instance segmentation mask)

All the Car-snow results are run on an extended validation set that I curated because I trust no one. To run on the extended validation set simply add the `--test-ext` flag to your command

### Editing Transformers
The transformer model I am using is ViT-B/16, which should be in [models/vit.py](models/vit.py). 

To run the editing tests for transformers with the attention loss:
`python finetune_vit.py --edit-type editing_scores --layernum 8.attn --test-ext --lr 0.01`

# Original Repo README
This repository contains the code and data for our paper:

**Editing a classifier by rewriting its prediction rules** <br>
*Shibani Santurkar\*, Dimitris Tsipras\*, Mahi Elango, David Bau, Antonio Torralba, Aleksander Madry* <br>
Paper: https://arxiv.org/abs/2112.01008 <br>

![](edit_examples.png)

```bibtex
    @InProceedings{santurkar2021editing,
        title={Editing a classifier by rewriting its prediction rules},
        author={Shibani Santurkar and Dimitris Tsipras and Mahalaxmi Elango and David Bau and Antonio Torralba and Aleksander Madry},
        year={2021},
        booktitle={Neural Information Processing Systems (NeurIPS)}
    }
```

## Getting started
You can start by cloning our repository and following the steps below. Parts of this codebase have been derived from the [GAN rewriting
repository](https://github.com/davidbau/rewriting) of Bau et al. 

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yaml
    ```

2. Download [model checkpoints](https://github.com/MadryLab/EditingClassifiers/releases/download/v1/checkpoints.tar.gz) and extract them in the current directory.

3. To instantiate CLIP
    ```
    git submodule init
    git submodule update

    ```
    
4. Replace `IMAGENET_PATH` in `helpers/classifier_helpers.py` with the path to the ImageNet dataset.
 
5. (If using synthetic examples) Download files [segmentations.tar.gz](https://github.com/MadryLab/EditingClassifiers/releases/download/v1/segmentations.tar.gz) and [styles.tar.gz](https://github.com/MadryLab/EditingClassifiers/releases/download/v1/styles.tar.gz) and extract them under `./data/synthetic`.

6. (If using synthetic examples) Run 
```
     python stylize.py --style_name [STYLE_FILE_NAME]
```
with the desired style file from `./data/synthetic/styles`. You could also use a custom style file if desired.

That's it! Now you can explore our editing methodology in various settings: [vehicles-on-snow](https://github.com/MadryLab/EditingClassifiers/blob/master/vehicles_on_snow.ipynb), [typographic attacks](https://github.com/MadryLab/EditingClassifiers/blob/master/typographic_attacks.ipynb) and [synthetic test cases](https://github.com/MadryLab/EditingClassifiers/blob/master/synthetic_test_cases.ipynb).

# Maintainers

* [Shibani Santurkar](https://twitter.com/ShibaniSan)
* [Dimitris Tsipras](https://twitter.com/tsiprasd)
