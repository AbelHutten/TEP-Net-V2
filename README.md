# TEP-Net-V2
Experiments with [TEP-Net (Laurent 2024)](arxiv.org/abs/2403.13094), with the goal of improving performance through architectural enhancements.

A large part of the code in this work comes from the repository that is associated with the publication, [train-ego-path-detection](https://github.com/irtrailenium/train-ego-path-detection), as this work is a continuation of that work. Please star [train-ego-path-detection](https://github.com/irtrailenium/train-ego-path-detection), and cite the original paper from Thomas Laurent, if you find his work useful:

```bibtex
@misc{laurent2024train,
      title={Train Ego-Path Detection on Railway Tracks Using End-to-End Deep Learning}, 
      author={Thomas Laurent},
      year={2024},
      eprint={2403.13094},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
The changes I made to his code are my own, and are not associated with the authors or publishers of the original work.

# Results Reproduction
<table style="table-layout: fixed; width: 100%;">
  <tr>
    <!-- First cell -->
    <td style="width: 50%; text-align: center; vertical-align: top;">
      <img src="assets/figure_8_Laurent_2024.png" alt="First Image" style="width:100%;">
      <br>
      <em>
        A reproduction of figure 8 from Laurent 2024. Results obtained on an NVIDIA Quadro RTX 6000 GPU, 
        with PyTorch v2.2.1 and TensorRT v8.6.1 (CUDA v12.1). Reproduced to simplify comparison. 
        Please refer to and cite the original paper when appropriate.
      </em>
    </td>
    <!-- Second cell -->
    <td style="width: 50%; text-align: center; vertical-align: top;">
      <img src="assets/model_evaluation_plot.png" alt="Second Image" style="width:100%;">
      <br>
      <em>
        Results of rerunning the training and evaluation. Results obtained on an NVIDIA RTX 4090 GPU 
        with PyTorch 2.5.0 and TensorRT v10.7.0 (CUDA 12.4).
      </em>
    </td>
  </tr>
</table>
