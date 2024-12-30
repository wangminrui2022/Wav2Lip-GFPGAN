py -0
where python
C:\Users\14312\AppData\Local\Programs\Python\Python310\python.exe

cd E:\wmr\Wav2Lip-GFPGAN
conda create --prefix ./venv python=3.10
conda env list
conda remove --name venv --all
conda activate E:\wmr\Wav2Lip-GFPGAN\venv

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia

https://developer.nvidia.com/cuda-toolkit-archive
https://pytorch.org/get-started/previous-versions/

conda activate E:\wmr\Wav2Lip-GFPGAN\venv
pip install E:\wmr\Wav2Lip-GFPGAN\whl\tb_nightly-2.19.0a20241229-py3-none-any.whl
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
pip install gdown
pip install scipy
pip install opencv-python
pip install librosa

pip install basicsr
pip install facexlib
pip install realesrgan


python -c "import torch; print(torch.__version__)"
nvidia-smi

pip install notebook
cd Wav2Lip

conda activate E:\wmr\Wav2Lip-GFPGAN\venv
jupyter notebook

gdown https://drive.google.com/uc?id=1fQtBSYEyuai9MjBOF8j7zZ4oQ9W2N64q --output {wav2lipPath}"/checkpoints/"


python E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\inference.py --checkpoint_path E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\checkpoints\wav2lip.pth --face E:\wmr\Wav2Lip-GFPGAN/inputs/kimk_7s_raw.mp4 --audio E:\wmr\Wav2Lip-GFPGAN/inputs/kim_audio.mp3 --outfile E:\wmr\Wav2Lip-GFPGAN/outputs/result.mp4

python E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\inference.py --checkpoint_path E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\checkpoints\wav2lip.pth --face E:\wmr\Wav2Lip-GFPGAN\inputs\run.mp4 --audio E:\wmr\Wav2Lip-GFPGAN\inputs\male.wav --outfile E:\wmr\Wav2Lip-GFPGAN\outputs\run.mp4

python E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\inference.py --checkpoint_path E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\checkpoints\wav2lip.pth --face E:\wmr\Wav2Lip-GFPGAN\inputs\run.mp4 --audio E:\wmr\Wav2Lip-GFPGAN\inputs\male.wav --outfile E:\wmr\Wav2Lip-GFPGAN\outputs\run.mp4

python E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\inference.py --checkpoint_path E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\checkpoints\wav2lip.pth --face E:\wmr\Wav2Lip-GFPGAN\inputs\run.mp4 --audio E:\wmr\Wav2Lip-GFPGAN\inputs\kimk_audio.mp3 --outfile E:\wmr\Wav2Lip-GFPGAN\outputs\run.mp4

python E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\inference.py --checkpoint_path E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\checkpoints\wav2lip.pth --face E:\wmr\Wav2Lip-GFPGAN\inputs\run.mp4 --audio E:\wmr\Wav2Lip-GFPGAN\inputs\sliant.mp3 --outfile E:\wmr\Wav2Lip-GFPGAN\outputs\run2.mp4

python E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\inference.py --checkpoint_path E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\checkpoints\wav2lip.pth --face E:\wmr\Wav2Lip-GFPGAN\inputs\run1.mp4 --audio E:\wmr\Wav2Lip-GFPGAN\inputs\sliant.mp3 --outfile E:\wmr\Wav2Lip-GFPGAN\outputs\run1.mp4

python E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\inference.py --checkpoint_path E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\checkpoints\wav2lip_gan.pth --face E:\wmr\Wav2Lip-GFPGAN\inputs\run1.mp4 --audio E:\wmr\Wav2Lip-GFPGAN\inputs\sliant.mp3 --outfile E:\wmr\Wav2Lip-GFPGAN\outputs\run1.mp4


python E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\inference.py --checkpoint_path E:\wmr\Wav2Lip-GFPGAN\Wav2Lip-master\checkpoints\wav2lip.pth --face E:\wmr\Wav2Lip-GFPGAN\inputs\run1.mp4 --audio E:\wmr\Wav2Lip-GFPGAN\inputs\male.wav --outfile E:\wmr\Wav2Lip-GFPGAN\outputs\run1.mp4



cd E:\wmr\Easy-Wav2Lip-8.3
conda create --prefix ./venv python=3.10
conda activate E:\wmr\Easy-Wav2Lip-8.3\venv
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt

https://www.youtube.com/watch?v=jArkTgAMA4g


cd E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master
conda create --prefix ./venv python=3.10
conda activate E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\venv
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install E:\wmr\Wav2Lip-GFPGAN\whl\tb_nightly-2.19.0a20241229-py3-none-any.whl
pip install -r requirements.txt
pip install realesrgan
pip install ffmpeg

python E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\inference_gfpgan.py -i E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\inputs\whole_imgs -o results -v 1.4 -s 2


https://github.com/iptop/GFPGAN-for-Video

python E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\src\video_enhance.py -i E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\inputs\video\close_output.mp4 -o E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\inputs\video\close_output.mp4-new.mp4

D:\ffmpeg\bin
ffmpeg -version

python E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\src\video_enhance.py -i E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\inputs\video\result.mp4 -o E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\inputs\video\result.mp4-new.mp4

python E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\src\video_enhance.py -i E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\inputs\video\final.mp4 -o E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\inputs\video\final.mp4-new.mp4

python E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\src\video_enhance.py -i E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\inputs\video\run.mp4 -o E:\wmr\Wav2Lip-GFPGAN\GFPGAN-master\inputs\video\run.mp4-new.mp4
