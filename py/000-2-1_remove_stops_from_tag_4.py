# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 18:14:52 2017

@author: konodera
"""
import os
print("""#==============================================================================
# START !!! {} PID: {}
#==============================================================================
""".format(__file__, os.getpid()))


import utils
stops = set(list(utils.stops)+["http","www","img","border","home","body","a",
"about","above","after","again","against","all","am","an","and","any","are",
"aren't","as","at","be","because","been","before","being","below","between",
"both","but","by","can't","cannot","could","couldn't","did","didn't","do",
"does","doesn't","doing","don't","down","during","each","few","for","from",
"further","had","hadn't","has","hasn't","have","haven't","having","he","he'd",
"he'll","he's","her","here","here's","hers","herself","him","himself","his",
"how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it",
"it's","its","itself","let's","me","more","most","mustn't","my","myself","no",
"nor","not","of","off","on","once","only","or","other","ought","our","ours",
"ourselves","out","over","own","same","shan't","she","she'd","she'll","she's",
"should","shouldn't","so","some","such","than","that","that's","the","their",
"theirs","them","themselves","then","there","there's","these","they","they'd",
"they'll","they're","they've","this","those","through","to","too","under","until",
"up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't",
"what","what's","when","when's""where","where's","which","while","who","who's",
"whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll",
"you're","you've","your","yours","yourself","yourselves","'s","n't"]+list(utils.kigou))


train, test = utils.load(4)


def rem_stop(wt):
    return [(w,t) for w,t in wt if w.lower() not in stops]

def main(df):
    df['q1_wt'] = df['q1_wt'].map(rem_stop)
    df['q2_wt'] = df['q2_wt'].map(rem_stop)
    return df


train = main(train)
test = main(test)


train.to_pickle('../input/mk/train_tag-stop.p')
test.to_pickle('../input/mk/test_tag-stop.p')

print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

