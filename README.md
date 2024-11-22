# CycleGAN-Finphot2Art

![UEF_art](assets/UEF_art.jpg) ![UEF_photo](assets/UEF_photo.jpg)

## First words
In this repository, I build and train a CycleGAN model from scratch, applying it to photos of landscapes from where I live in Joensuu, Finland. The goal is to create some amazing artwork.

## Usage
<details open>
<summary>Requirements</summary>
<ul>
<li>Python3</li>
<li>tqdm</li>
<li>numpy</li>
<li>matplotlib</li>
<li>seaborn</li>
<li>opencv-python</li>
<li>torch</li>
<li>torchvision</li>
</ul>
</details>

<details open>
<summary>Install</summary>

First, clone the repository.
```bash
git clone https://github.com/HoangPham3003/CycleGAN-Finphot2Art.git
cd CycleGAN-Finphot2Art
```
Second, create and activate the python environment.
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Finally, install requirements.
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
</details>

<details open>
<summary>Train</summary>

Default parameters <br>
To start training with the default parameters, run: <br>
``` bash
python train.py
```

Custom parameters <br>
To customize training, you can adjust the parameters as follows: <br>
``` bash
python train.py -pt CycleGAN.pt -lr 0.0002 -ep 20 -bs 1 -ds 200 -ts 256 -d cuda -s True 
```
For a detailed explanation of the training parameters, refer to the [train.py](https://github.com/HoangPham3003/CycleGAN-Finphot2Art/blob/main/train.py)
</details>

<details open>
<summary>Infer</summary>
To generate a new artwork from a photo:
``` bash
python infer.py -pt CycleGAN.pt -ip img1.jpg -sd inference -d cuda
```
For a detailed explanation of the inference parameters, refer to the [infer](https://github.com/HoangPham3003/CycleGAN-Finphot2Art/blob/main/infer)
</details>



