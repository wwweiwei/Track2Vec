# Track2Vec - Fairness Music Recommendation with a GPU-Free Customizable-Driven Framework
:bulb: This is the official code of team wwweiwei to the [EvalRS Data Challenge](https://github.com/RecList/evalRS-CIKM-2022). We won the fouth place. For more details, please refer to our [paper](http://arxiv.org/abs/2210.16590) and [brief introduction](https://medium.com/@wwweiwei/cikm-2022-track2vec-fairness-music-recommendation-with-a-gpu-free-customizable-driven-framework-d1959194bfc1) in our blog. <br> 

## Usage
### Setup
- Build environment
    ```
    pip install -r /path/to/requirements.txt
    ```
- Place your `upload.env` in the root folder.

### Run script
```
python submission.py
```
- Notes: Our proposed metric MR-ITF will automatically report in the corresponding json file with other standard metric.
## Introduction
- Proposed Framework: Track2Vec
<img width="624" alt="Track2Vec Framework" src="images/Track2Vec_framework.jpg">

- Proposed Fairness Metric: Miss Rate - Inverse Ground Truth Frequency (MR-ITF)
<img width="324" alt="MR_ITF_equation" src="images/MR_ITF_equation.png">


## Citation
If you find our work is relevant to your research, please cite:
```
@inproceedings{DBLP:conf/cikm/DuWP22,
  author    = {Wei{-}Wei Du and
               Wei{-}Yao Wang and
               Wen{-}Chih Peng},
  title     = {Track2Vec: fairness music recommendation with a GPU-free customizable-driven
               framework},
  booktitle = {{CIKM} Workshops},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {3318},
  publisher = {CEUR-WS.org},
  year      = {2022}
}
```
