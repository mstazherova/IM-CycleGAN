# IM-CycleGAN
Individual Module Project, University of Potsdam

The purpose of this project was to apply cycle-consistent adversatial networks proposed by Zhu et al. [1] to the face attribute manipulation problem.

We investigate the use of unpaired image-to-image translation using [CycleGAN](https://junyanz.github.io/CycleGAN/) to the task of eyeglasses removal from faces along with the reverse task of adding eyeglasses to facial images.

Final version can be found in `keras` folder. All implementation details and model architecture can also be found in the project paper (report_cycleGAN.pdf).

## Data
Data folders are not being uploaded to GitHub due to size issues and a little bit of privacy.

Sets:
* Eyeglasses: 1777 training images

* No-eyeglasses: 1687 training images


## References
[1] Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). [Unpaired image-to-image translation using cycle-consistent adversarial networks.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)
