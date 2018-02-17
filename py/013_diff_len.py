# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:20:09 2017

@author: Kazuki
"""
import utils
import multiprocessing as mp
total_proc = 2

w2v_google = utils.load_vec('w2v-google')
w2v_quora = utils.load_vec('w2v-quora-0')



train, test = utils.load(8, 1)

#==============================================================================
# def
#==============================================================================
def get_diff_len(q1, q2, diff, THERESHOLD):
    q = q1.split() + q2.split()
    diff_bk = diff[:]
    for w in diff:
        q_ = {w_ for w_ in q if w != w_}
        for w_ in q_:
            try:
                if w2v_quora.similarity(w, w_) > THERESHOLD:
#                    print(w, w_, w2v_quora.similarity(w, w_))
                    diff_bk.remove(w)
                    break
            except:
                pass
    return len(diff_bk)


def main(p):
    """
    df = train.sample(999)
    df_ = df[['id']]
    """
    
    if p==0:
        df = train
        df_ = df[['id']]
        fname = '../feature/train_f013.csv.gz'
    elif p==1:
        df = test
        df_ = df[['test_id']]
        fname = '../feature/test_f013.csv.gz'
    
    df_['q1_set_diff_len'] = df.q1_set_diff.map(len)
    df_['q2_set_diff_len'] = df.q2_set_diff.map(len)
    df_['set_diff_len'] = df.set_diff.map(len)
    
    df_['set_diff_len_w2v09'] = df.apply(lambda x: get_diff_len(x['q1'], x['q2'], x['set_diff'], 0.9), axis=1)
    df_['set_diff_len_w2v08'] = df.apply(lambda x: get_diff_len(x['q1'], x['q2'], x['set_diff'], 0.8), axis=1)
    df_['set_diff_len_w2v07'] = df.apply(lambda x: get_diff_len(x['q1'], x['q2'], x['set_diff'], 0.7), axis=1)
    df_['set_diff_len_w2v06'] = df.apply(lambda x: get_diff_len(x['q1'], x['q2'], x['set_diff'], 0.6), axis=1)
    df_['set_diff_len_w2v05'] = df.apply(lambda x: get_diff_len(x['q1'], x['q2'], x['set_diff'], 0.5), axis=1)
    df_['set_diff_len_w2v04'] = df.apply(lambda x: get_diff_len(x['q1'], x['q2'], x['set_diff'], 0.4), axis=1)
    df_['set_diff_len_w2v03'] = df.apply(lambda x: get_diff_len(x['q1'], x['q2'], x['set_diff'], 0.3), axis=1)
    
    # ratio
    df_['set_diff_len_w2v09_ratio'] = df_['set_diff_len_w2v09'] / df_['set_diff_len']
    df_['set_diff_len_w2v08_ratio'] = df_['set_diff_len_w2v08'] / df_['set_diff_len']
    df_['set_diff_len_w2v07_ratio'] = df_['set_diff_len_w2v07'] / df_['set_diff_len']
    df_['set_diff_len_w2v06_ratio'] = df_['set_diff_len_w2v06'] / df_['set_diff_len']
    df_['set_diff_len_w2v05_ratio'] = df_['set_diff_len_w2v05'] / df_['set_diff_len']
    df_['set_diff_len_w2v04_ratio'] = df_['set_diff_len_w2v04'] / df_['set_diff_len']
    df_['set_diff_len_w2v03_ratio'] = df_['set_diff_len_w2v03'] / df_['set_diff_len']
    
    df_.to_csv(fname, compression='gzip', index=0)
    
    return

#==============================================================================

pool = mp.Pool(total_proc)
callback = pool.map(main, range(total_proc))




print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

