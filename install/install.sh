conda env list &&
conda deactivate &&
conda remove -n PCCLS --all -y &&
conda create -n PCCLS -y python=3.8 numpy=1.23.4 numba &&
conda activate PCCLS &&
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -y &&
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package &&
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip &&
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple &&
pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html &&
pip install -r requirements.txt