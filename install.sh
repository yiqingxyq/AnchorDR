pip install sentencepiece
pip install datasets
pip install faiss-gpu
pip install tensorboard
pip install deepspeed==0.6.5

# requirements for FairseqT5
pip install omegaconf==2.0.6
pip install setuptools==59.5.0
pip install hydra-core==1.0.7
pip install scipy==1.7.3
pip install sacrebleu==2.3.1
pip install editdistance
pip install scikit-learn

# install transformers 
cd ${CODE_DIR}/transformers
pip install -e .

cd $EVAL_DIR
make