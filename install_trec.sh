# install trec_eval
cp ${CODE_DIR}/metrics/trec/trec_eval-9.0.7.tar.gz ${BASE_DIR}

cd ${BASE_DIR}
tar -xvf trec_eval-9.0.7.tar.gz
mv trec_eval-9.0.7 trec_eval
cd trec_eval
make
