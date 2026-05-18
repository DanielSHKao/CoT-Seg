# CoT-Seg: Rethinking Segmentation with Chain-of-Thought Reasoning and Self-Correction
> [**CoT-Seg: Rethinking Segmentation with Chain-of-Thought Reasoning and Self-Correction**](https://arxiv.org/pdf/2601.17420)     
> Shiu-hong Kao, Chak Ho Huang, Huaiqian Liu, Yu-Wing Tai, Chi-Keung Tang  
> ICLR 2026 Workshop on AI with Recursive Self-Improvement (RSI)  
> [Project page](https://danielshkao.github.io/cot-seg.html)

<img width="800" alt="image" src="https://github.com/user-attachments/assets/04dcfe23-e97d-40b6-a0ea-cdbbca8f9cb7" />

CoT-Seg is a a training-free framework that rethinks reasoning segmentation by combining chain-of-thought reasoning with self-correction. 
CoT-Seg leverages the inherent reasoning ability of pre-trained MLLMs
(e.g., GPT-4o) to decompose queries into meta-instructions, extract fine-grained
semantics from images, and identify target objects even under implicit or complex
prompts. Crucially, CoT-Seg incorporates a self-correction stage: the model evaluates its own segmentation against the original query and reasoning trace, identifies mismatches, and iteratively refines the mask.

---
## Installation
Our code is tested on Ubuntu 22.04 with python 3.12

Installing requirements
```commandline
pip install -r requirements.txt
```
Flash attention
```commandline
pip install flash_attn==2.7.4.post1 --no-build-isolation
```

## Model Downloads
1. Download the SAM Checkpoint, we used SamHQ2 ([sam2.1_hq_hiera_large](https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt?download=true)).
2. Download the Reasoning Segmentation Model, we mainly used [VisionReasoner](https://github.com/JIA-Lab-research/VisionReasoner).

## Inference
To run inference, please first set the configuration for OpenAI API in `config/openai.yaml`,
then you can run:

```commandline
python inference.py 
--reasoner_model_path [path] \
--segmentation_model_path [path] \
--prompt [prompt] \
--image_path [path] \
--output_path [path] \
--reseg_rounds 2 \
--mask_threshold 0.5 \
--use_rag False
``` 

## Demo
<img width="700" alt="image" src="https://github.com/user-attachments/assets/77e00d7e-8c3a-4e69-a412-9b8a73868bc5" />

View more examples in our [project page](https://danielshkao.github.io/cot-seg.html).

## Dataset
We evaluated CoT-Seg on [ReasonSeg](https://github.com/JIA-Lab-research/LISA#Dataset), [ReasonSeg-Hard](https://github.com/DanielSHKao/CoT-Seg/blob/main/dataset), and [RefCOCO](https://github.com/lichengunc/refer). Please proceed to the official websites to download the data.

## Citation
If you find this repository helpful, please consider citing:
```
@article{kao2026cot,
  title={CoT-Seg: Rethinking Segmentation with Chain-of-Thought Reasoning and Self-Correction},
  author={Kao, Shiu-hong and Huang, Chak Ho and Liu, Huaiqian and Tai, Yu-Wing and Tang, Chi-Keung},
  journal={arXiv preprint arXiv:2601.17420},
  year={2026}
}
```
