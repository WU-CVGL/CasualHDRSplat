<h2 align="center">CasualHDRSplat: Robust High Dynamic Range 3D Gaussian Splatting from
Casually Captured Videos </br> </br>
ArXiv 2025</h2>
<p align="center">
    <a href="https://github.com/TeaDrinkingGong">Shucheng Gong</a><sup>1,2*</sup> &emsp;&emsp;
    <a href="https://github.com/LingzheZhao">Lingzhe Zhao</a><sup>1*</sup> &emsp;&emsp;
    <a href="https://akawincent.github.io">Wenpu Li</a><sup>1*</sup> &emsp;&emsp;
    Xiang Liu<sup>1,3</sup> &emsp;&emsp; </br>
    Yin Zhang<sup>1,4</sup> &emsp;&emsp;
    Shiyu Zhao<sup>1</sup> &emsp;&emsp;
    Hong Xie<sup>2</sup> &emsp;&emsp;
    <a href="https://ethliup.github.io/">Peidong Liu</a><sup>1†</sup>
</p>

<p align="center">
    <sup>*</sup>equal contribution &emsp;&emsp; <sup>†</sup> denotes corresponding author.
</p>

<p align="center">
    <sup>1</sup>Westlake University &emsp;&emsp;
    <sup>2</sup>Wuhan University &emsp;&emsp;
    <sup>3</sup>ETH Zürich &emsp;&emsp;
    <sup>4</sup>Zhejiang University &emsp;&emsp;
</p>

<hr>

<h5 align="center">

<!-- [![arXiv](https://img.shields.io/badge/Arxiv-2407.02174-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2407.02174v3)
[![pdf](https://img.shields.io/badge/PDF-Paper-orange.svg?logo=GoogleDocs)](./doc/2024_ECCV_BeNeRF_camera_ready_paper.pdf) 
[![pdf](https://img.shields.io/badge/PDF-Supplementary-orange.svg?logo=GoogleDocs)](./doc/2024_ECCV_BeNeRF_camera_ready_supplementary.pdf) 
[![pdf](https://img.shields.io/badge/PDF-Poster-orange.svg?logo=GoogleDocs)](https://akawincent.github.io/BeNeRF/demo/Poster.pdf) 
[![Home Page](https://img.shields.io/badge/GitHubPages-ProjectPage-blue.svg?logo=GitHubPages)](https://akawincent.github.io/BeNeRF/)
[![Paper With Code](https://img.shields.io/badge/Website-PaperwithCode-yellow.svg?logo=paperswithcode)](https://paperswithcode.com/paper/benerf-neural-radiance-fields-from-a-single)  
[![Dataset](https://img.shields.io/badge/OneDrive-Dataset-green.svg?logo=ProtonDrive)](https://westlakeu-my.sharepoint.com/:f:/g/personal/cvgl_westlake_edu_cn/EjZNs8MwoXBDqT61v_j5V3EBIoKb8dG9KlYtYmLxcNJG_Q?e=AFXeUB) -->
![GitHub Repo stars](https://img.shields.io/github/stars/WU-CVGL/CasualHDRSplat)

</h5>
<p align="center">
    <img src="./assets/teaser.png" alt="Pipeline" style="width:75%; height:auto;">
</p>

> a) Our method can reconstruct 3D HDR scenes from videos casually captured with auto-exposure enabled. b) Our approach achieves superior rendering quality compared to methods like Gaussian-W and HDR-Plenoxels. c) After 3D HDR reconstruction, we can not only synthesize novel view, but also perform various downstream tasks, such as 1) HDR exposure editing, 2) Image deblurring.


## 📋 Pipeline

<p align="center">
    <img src="./assets/pipeline.png" alt="Pipeline" style="width:75%; height:auto;">
</p>

<div>
Given a casually captured video with auto exposure, camera motion blur, and significant exposure time changes, we train 3DGS to reconstruct an HDR scene. We design a unified model based on the physical image formation process, integrating camera motion blur and exposure-induced brightness variations. This allows for the joint estimation of camera motion, exposure time, and camera response curve while reconstructing the HDR scene. After training, our method can sharpen the train images and render HDR and LDR images from specified poses.
</div>

## 🛠️  Still working on....

