- pip freeze > requirements.txt
- pip install -r requirements.txt


- conda create -n py3 --clone root
- conda create -n py3 python=3.8.10 numpy scipy pandas cython matplotlib
- conda remove -n xx --all 
- conda env list 
- conda activate xxx
- conda deactivate


- curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
- python3 get-pip.py --force-reinstall



- pip install virtualenv
- python -m venv venv
- source venv/bin/activate 
- deactivate


- bash ~/anaconda3.sh -b -p $HOME/anaconda3
- export PATH=$HOME/anaconda3/bin:$HOME/anaconda3/condabin:$PATH
- source ~/.bashrc
