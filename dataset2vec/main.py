from sklearn.datasets import load_svmlight_file
from dataset2vec.d2v import D2V
from dataset2vec.runner import Runner
import pandas as pd
import sklearn

def main(config,dataset,seed):
    X_train, y_train = load_svmlight_file(dataset)
    mat = X_train.todense()
    X = pd.DataFrame(mat)
    X.columns = range(len(X.columns))
    y = pd.DataFrame(y_train)
    y.columns = ['target']
    ds = X_train, y_train
    lfp = D2V(config=config)
    optimizer = Runner(config=config,dataset=ds,model=lfp)
    optimizer.run(dataset=dataset)
    phi = optimizer.summarize()
    return phi
    
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--dataset', help='Which Configuration', type=str,default='toy')
    parser.add_argument('--seed', help='Which Configuration', type=int,default=0)
    args = parser.parse_args()
    from dataset2vec.config import config
    main(config=config,dataset=args.dataset,seed=args.seed)
