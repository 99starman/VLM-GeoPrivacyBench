<h1>Do Vision-Language Models Respect Contextual Integrity in Location Disclosure?</h1>


<a href='https://arxiv.org/abs/2602.05023'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://huggingface.co/datasets/RayY/VLM-GeoPrivacyBench'><img src='https://img.shields.io/badge/🤗-Dataset-blue'></a>
</a>
 

<h2 id="updates">🔥 Updates</h2>

📆[2026-01-26] 🎈 Our paper is accepted by ICLR 2026! 🎈

📆[2025-10-25] 🎈 Our code and dataset are released! 🎈


<h2 id="SALAD-Bench">🎉 Introduction</h2>

<div align="center">
<img src="./doc_images/teaser.png" width="900" alt="Intro_img">
</div>

Vision-language models (VLMs) have demonstrated strong performance in image geolocation, a capability further sharpened by frontier multimodal large reasoning models (MLRMs). This poses a significant privacy risk, as these widely accessible models can be exploited to infer sensitive locations from casually shared photos, often at street-level precision, potentially surpassing the level of detail the sharer consented or intended to disclose. While recent work has proposed applying a blanket restriction on geolocation disclosure to combat this risk, these measures fail to distinguish valid geolocation uses from malicious behavior. Instead, VLMs should maintain contextual integrity by reasoning about elements within an image to determine the appropriate level of information disclosure, balancing privacy and utility. To evaluate how well models respect contextual integrity, we introduce VLM-GeoPrivacy, a benchmark that challenges VLMs to interpret latent social norms and contextual cues in real-world images and determine the appropriate level of location disclosure. Our evaluation of 14 leading VLMs shows that, despite their ability to precisely geolocate images, the models are poorly aligned with human privacy expectations. They often over-disclose in sensitive contexts and are vulnerable to prompt-based attacks. Our results call for new design principles in multimodal systems to incorporate context-conditioned privacy reasoning.


<h2 id="dataset">⚙️ Dataset</h2>

Our dataset consists of 1,200 real-world images richly annotated with context, sharing intent, and expected granularity. The images were carefully curated from general geolocation datasets including [YFCC4k](https://github.com/lugiavn/revisiting-im2gps), [YFCC26k](https://github.com/TIBHannover/GeoEstimation), [YFCC100M-OpenAI](https://huggingface.co/datasets/dalle-mini/YFCC100M_OpenAI_subset), [Im2GPS3k](https://github.com/lugiavn/revisiting-im2gps), and [GPTGeoChat](https://github.com/ethanm88/GPTGeoChat). We follow their original licenses for the images, and will not host or directly distribute them due to copyright considerations. Instead, we provide a script for the users to retrieve the images from source datasets. Please refer to `data/image_download_instructions.md` for specific instructions.

<h2 id="quick-start">🚀 Usage</h2>

1. Follow the instructions from `data/image_download_instructions.md`, download and unzip the images to `data/images`.

2. Set up environment: 

    First, create conda/virtual environment, and install packages, e.g.
    ```
    conda create -n geoprivbench python=3.12
    conda activate geoprivbench
    cd VLM-GeoPrivacyBench
    pip install -r requirements.txt
    ```

    Then, create a `.env` in the project root with the following evaluation api keys:
    ```
    AZURE_API_KEY = 
    AZURE_OPENAI_ENDPOINT = 
    AZURE_INFERENCE_SDK_ENDPOINT =
    GEMINI_API_KEY=
    CLAUDE_API_KEY=
    GOOGLE_GEOCODING_API_KEY=
    OPENAI_API_KEY (optional; if MS Azure not available)
    ``` 

3. Optionally, (re)run model generation with `sbatch scripts/gen.sh` (GPUs required) or `source scripts/gen_api.sh` (API-based models). Results are saved in `evaluation/main`, where we already include the model generation results used in our experiments.

4. Evaluate the model generation with `source scripts/eval.sh` (various settings specified in the script).
