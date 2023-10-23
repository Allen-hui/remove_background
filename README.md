# Background Removal with Deep Learning

This repository show the code to remove the background of the pictures using
the [U2Net](https://arxiv.org/pdf/2005.09007.pdf)
and [DIS](https://github.com/xuebinqin/DIS/blob/main/DIS5K-Dataset-Terms-of-Use.pdf) pre-trained model.

The application has three simple functions:

1. Remove the background, producing a transparent PNG file.

2. Change the background by another picture.

3. Combine the image and multiple backgrounds to augment the dataset.

### Endpoint available
| Endpoint | Description
| --- | ---
| http://localhost:8000/ |  Front-end to perform background remove.
| http://localhost:8000/augmentation |  Front-end to perform augment images.

### Install
1. Clone this repository
```bash
git clone https://github.com/Allen-hui/remove_background.git
cd remove_background
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download the pre-trained model
```bash
gdown --id 1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ -O ./ckpt/u2net.pth
gdown --id 1nV57qKuy--d5u1yvkng9aXW1KS4sOpOi -O ./ckpt/isnet-general-use.pth
```

4. Start web-application
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### References
U2Net: [https://github.com/xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)

DIS: [https://github.com/xuebinqin/DIS](https://github.com/xuebinqin/DIS)
