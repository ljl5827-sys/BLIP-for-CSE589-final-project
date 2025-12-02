# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation  
### (Modified Version for CSE589 Final Project — Linlin Li, Penn State University)

This repository contains my CSE589 final project, where I reproduce BLIP’s COCO 2014 image–text retrieval task under limited single-GPU computation.  
The project is based on the official BLIP implementation from Salesforce Research, with several required modifications to enable single-GPU execution, small-scale dataset debugging, and custom hyperparameter experiments.

Below is a list of all modifications made for this project:

---

## 🔧 Modifications Made for This Project

### 1. Dataset Reduction (`import_json.py`)
To speed up debugging, I created a script that extracts 100 caption entries from the COCO Karpathy test set:

```python
import json
src = r".../coco_karpathy_test.json"
dst = r".../coco_karpathy_test_small.json"

with open(src, "r") as f:
    data = json.load(f)

small_data = data[:100]
with open(dst, "w") as f:
    json.dump(small_data, f)
```

This was used only for debugging.  
The final results in the report use the full 500-image test set.

---

### 2. Modified `configs/retrieval_coco.yaml`

**Updated dataset paths:**
```yaml
image_root: "D:\\CSE589\\Linlin_final\\dataset\\images"
ann_root:   "D:\\CSE589\\Linlin_final\\dataset\\annotation"
pretrained: "D:\\CSE589\\Linlin_final\\BLIP-main\\checkpoints\\model_base_retrieval_coco.pth"
```

**Reduced training batch size** for single GPU:
```yaml
batch_size_train: 4
batch_size_test: 8
```

These results are analyzed in the project report.

---

### 3. Modified `train_retrieval.py`

**(A) Disabled multi-GPU barriers to avoid single-GPU hanging:**

Original:
```python
if args.distributed and dist.is_initialized():
    dist.barrier()
```

Modified:
```python
if args.distributed and dist.is_initialized():
    dist.barrier()
# torch.cuda.empty_cache()
```

**(B) Changed defaults to use COCO:**
```python
parser.add_argument('--config', default='./configs/retrieval_coco.yaml')
parser.add_argument('--output_dir', default='output/Retrieval_coco')
```

**(C) Fixed YAML saving for Windows:**
```python
with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
    yaml.dump(config, f)
```

---

### 4. Modified `blip_retrieval.py`

The original implementation uses `torch.distributed.all_gather`, which breaks when only one GPU is available.

Original:
```python
tensors_gather = [...]
torch.distributed.all_gather(...)
output = torch.cat(..., dim=0)
return output
```

Modified to:
```python
@torch.no_grad()
def concat_all_gather(tensor):
    return tensor
```

Reason:  
On a single GPU, world size = 1, so concatenation and all_gather are unnecessary and cause size mismatches or runtime errors. Returning the tensor directly restores correct evaluation behavior.

---


<img src="BLIP.gif" width="700">

This is the PyTorch code of the <a href="https://arxiv.org/abs/2201.12086">BLIP paper</a> [[blog](https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/)]. The code has been tested on PyTorch 1.10.
To install the dependencies, run <pre/>pip install -r requirements.txt</pre> 

Catalog:
- [x] Inference demo
- [x] Pre-trained and finetuned checkpoints
- [x] Finetuning code for Image-Text Retrieval, Image Captioning, VQA, and NLVR2
- [x] Pre-training code
- [x] Zero-shot video-text retrieval
- [x] Download of bootstrapped pre-training datasets 


### Inference demo:
Run our interactive demo using [Colab notebook](https://colab.research.google.com/github/salesforce/BLIP/blob/main/demo.ipynb) (no GPU needed).
The demo includes code for: 
1. Image captioning
2. Open-ended visual question answering
3. Multimodal / unimodal feature extraction
4. Image-text matching

Try out the [Web demo](https://huggingface.co/spaces/Salesforce/BLIP), integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). 

Replicate web demo and Docker image is also available at [![Replicate](https://replicate.com/salesforce/blip/badge)](https://replicate.com/salesforce/blip)

### Pre-trained checkpoints:
Num. pre-train images | BLIP w/ ViT-B | BLIP w/ ViT-B and CapFilt-L | BLIP w/ ViT-L 
--- | :---: | :---: | :---: 
14M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth">Download</a>| - | -
129M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth">Download</a> | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth">Download</a>

### Finetuned checkpoints:
Task | BLIP w/ ViT-B | BLIP w/ ViT-B and CapFilt-L | BLIP w/ ViT-L 
--- | :---: | :---: | :---:
Image-Text Retrieval (COCO) | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth">Download</a>| - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth">Download</a>
Image-Text Retrieval (Flickr30k) | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth">Download</a>|  - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_flickr.pth">Download</a>
Image Captioning (COCO) | - | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth">Download</a> | 
VQA | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth">Download</a>| <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth">Download</a> | - 
NLVR2 | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_nlvr.pth">Download</a>| - | - 


### Image-Text Retrieval:
1. Download COCO and Flickr30k datasets from the original websites, and set 'image_root' in configs/retrieval_{dataset}.yaml accordingly.
2. To evaluate the finetuned BLIP model on COCO, run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco \
--evaluate</pre> 
3. To finetune the pre-trained checkpoint using 8 A100 GPUs, first set 'pretrained' in configs/retrieval_coco.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_retrieval.py \
--config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco </pre> 

### Image-Text Captioning:
1. Download COCO and NoCaps datasets from the original websites, and set 'image_root' in configs/caption_coco.yaml and configs/nocaps.yaml accordingly.
2. To evaluate the finetuned BLIP model on COCO, run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_caption.py --evaluate</pre> 
3. To evaluate the finetuned BLIP model on NoCaps, generate results with: (evaluation needs to be performed on official server)
<pre>python -m torch.distributed.run --nproc_per_node=8 eval_nocaps.py </pre> 
4. To finetune the pre-trained checkpoint using 8 A100 GPUs, first set 'pretrained' in configs/caption_coco.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=8 train_caption.py </pre> 

### VQA:
1. Download VQA v2 dataset and Visual Genome dataset from the original websites, and set 'vqa_root' and 'vg_root' in configs/vqa.yaml.
2. To evaluate the finetuned BLIP model, generate results with: (evaluation needs to be performed on official server)
<pre>python -m torch.distributed.run --nproc_per_node=8 train_vqa.py --evaluate</pre> 
3. To finetune the pre-trained checkpoint using 16 A100 GPUs, first set 'pretrained' in configs/vqa.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=16 train_vqa.py </pre> 

### NLVR2:
1. Download NLVR2 dataset from the original websites, and set 'image_root' in configs/nlvr.yaml.
2. To evaluate the finetuned BLIP model, run
<pre>python -m torch.distributed.run --nproc_per_node=8 train_nlvr.py --evaluate</pre> 
3. To finetune the pre-trained checkpoint using 16 A100 GPUs, first set 'pretrained' in configs/nlvr.yaml as "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth". Then run:
<pre>python -m torch.distributed.run --nproc_per_node=16 train_nlvr.py </pre> 

### Finetune with ViT-L:
In order to finetune a model with ViT-L, simply change the config file to set 'vit' as large. Batch size and learning rate may also need to be adjusted accordingly (please see the paper's appendix for hyper-parameter details). <a href="https://github.com/facebookresearch/fairscale">Gradient checkpoint</a> can also be activated in the config file to reduce GPU memory usage. 

### Pre-train:
1. Prepare training json files where each json file contains a list. Each item in the list is a dictonary with two key-value pairs: {'image': path_of_image, 'caption': text_of_image}. 
2. In configs/pretrain.yaml, set 'train_file' as the paths for the json files .
3. Pre-train the model using 8 A100 GPUs:
<pre>python -m torch.distributed.run --nproc_per_node=8 pretrain.py --config ./configs/Pretrain.yaml --output_dir output/Pretrain </pre> 

### Zero-shot video-text retrieval:
1. Download MSRVTT dataset following the instructions from https://github.com/salesforce/ALPRO, and set 'video_root' accordingly in configs/retrieval_msrvtt.yaml.
2. Install [decord](https://github.com/dmlc/decord) with <pre>pip install decord</pre> 
3. To perform zero-shot evaluation, run
<pre>python -m torch.distributed.run --nproc_per_node=8 eval_retrieval_video.py</pre> 

### Pre-training datasets download:
We provide bootstrapped pre-training datasets as json files. Each json file contains a list. Each item in the list is a dictonary with two key-value pairs: {'url': url_of_image, 'caption': text_of_image}. 

Image source | Filtered web caption | Filtered synthetic caption by ViT-B | Filtered synthetic caption by ViT-L
--- | :---: | :---: | :---:
CC3M+CC12M+SBU |  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_filtered.json">Download</a>|  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_synthetic_filtered.json">Download</a>|  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/ccs_synthetic_filtered_large.json">Download</a>
LAION115M | <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/laion_filtered.json">Download</a>|  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/laion_synthetic_filtered.json">Download</a>|  <a href="https://storage.googleapis.com/sfr-vision-language-research/BLIP/datasets/laion_synthetic_filtered_large.json">Download</a>

### Citation
If you find this code to be useful for your research, please consider citing.
<pre>
@inproceedings{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation}, 
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      booktitle={ICML},
}</pre>

### Acknowledgement
The implementation of BLIP relies on resources from <a href="https://github.com/salesforce/ALBEF">ALBEF</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and <a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm">timm</a>. We thank the original authors for their open-sourcing.
