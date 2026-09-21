# Image credits

| file | author | license | source |
|---|---|---|---|
| `lecture01/mark1-perceptron.jpg` | US Navy (National Museum of the U.S. Navy), photo 330-PSA-80-60 (USN 710739), released 24 June 1960 | Public domain | https://commons.wikimedia.org/wiki/File:330-PSA-80-60_(USN_710739)_(20897323365).jpg |
| `lecture06/chelsea.jpg` | Stefan van der Walt (“Chelsea the cat”, scikit-image sample data) | CC0 | `skimage.data.chelsea()` |
| `lecture06/bison.jpg` | WikiImages (Pixabay) | Pixabay Content License | taken from the 2024/25 slides (“Image by WikiImages from Pixabay”), resized to 800 px |
| `lecture05/lab-75.jpg`, `lab-83.jpg`, `lab-84.jpg` | MIMUW robot learning lab (Mateusz Wyszyński), own photos from the 2025/26 lecture deck | course authors | — |
| `lecture05/gen-robots.jpg` | generated for this deck with Z-Image-Turbo (Tongyi-MAI, Apache 2.0; 4-bit weights `mflux-community/z-image-turbo-mflux-q4` run with mflux), prompt "two black robot arms on a wooden table folding a cardboard box, bright university robotics lab, photorealistic", seed 42 | course authors | — |
| `lecture05/pi0-fig3.png` | Figure 3 of Black et al., “π0: A Vision-Language-Action Flow Model for General Robot Control”, Physical Intelligence, 2024 (cropped from the arXiv PDF; the coloured frames are added in the slide) | the authors | https://arxiv.org/abs/2410.24164 |
| `lecture05/pi0-robot.mp4`, `pi0-robot-poster.jpg` | the π0 action-chunk denoising animation from the 2025/26 lecture deck (Mateusz Wyszyński), converted from GIF to H.264 | course authors | — |
| `lecture07/pears.jpg`, `apple.jpg` | Tracy Lundgren; Michal Jarmoluk (Pixabay) | Pixabay Content License | taken from the 2025/26 slides (“Image by … from Pixabay”), resized to 960 / 640 px |
| `lecture07/street.jpg` | Unsplash photo on Wikimedia Commons (“Pedestrians on a crosswalk”, CC0), resized to 960 px | CC0 | https://commons.wikimedia.org/wiki/File:Pedestrians_on_a_crosswalk_(Unsplash).jpg |
| `lecture07/crowd.jpg` | Unsplash photo on Wikimedia Commons (“Crosswalk in Sapporo”, CC0), cropped | CC0 | https://commons.wikimedia.org/wiki/File:Crosswalk_in_Sapporo_(Unsplash).jpg |
| `lecture07/distill/*.png` | Odena, Dumoulin & Olah, “Deconvolution and Checkerboard Artifacts”, Distill 2016 (their own figures, not the reused GAN samples) | CC BY 2.0 | https://distill.pub/2016/deconv-checkerboard/ |

Resized to at most 800 px. All other figures are drawn in code (`shared/widgets/`), inline SVG, or charts of our own experiments. The flow-matching models in lecture 5 are trained by `experiments/flow2d_train.mjs` (weights in `lecture05/data/`). The Nakkiran et al. figure is redrawn from digitized values (see `lecture01/data/nakkiran2019_fig1_digitized.json`).

Lecture 6: `shuffled_*.png` (MNIST: LeCun, Cortes & Burges; CIFAR-10: Krizhevsky 2009), `alexnet_conv1.png` and `fmap_*.png` are produced by our scripts in `experiments/` from the datasets and torchvision's pretrained weights.

Lecture 6, ImageNet slide: `lecture06/imagenet/*.jpg` are images from ImageNetV2 (Recht et al., 2019; Flickr photos collected with the ImageNet procedure), centre-cropped to 192 px by `experiments/imagenet_samples.py`; credited on the slide like the MNIST / CIFAR-10 samples.

Lecture 7: everything in `lecture07/data/` (segmentation maps, detections, proposals, masks, SAM masks, OWLv2 boxes) is produced by our scripts in `experiments/` (`segmentation.py`, `detection.py`, `street_boxes.py`, `two_stage.py`, `mask_head.py`, `pear_softmax.py`, `sam_points.py`, `sam_modes.py`, `open_vocab.py`) from the photos above, with torchvision's pretrained COCO / VOC models, Meta's SAM ViT-B checkpoint (Apache 2.0) and Google's OWLv2 (Apache 2.0).
