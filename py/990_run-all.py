"""
#==============================================================================
# 000
#==============================================================================
nohup python -u 000-0_mk_data.py &
input:  ../input/train.csv.zip
        ../input/test.csv.zip
output: ../input/mk/train_prep.csv.gz #0
        ../input/mk/test_prep.csv.gz
        ../input/mk/train_stem.csv.gz #1
        ../input/mk/test_stem.csv.gz
        ../input/mk/train_stop.csv.gz #2
        ../input/mk/test_stop.csv.gz
        ../input/mk/train_stop-stem.csv.gz #3
        ../input/mk/test_stop-stem.csv.gz


nohup python -u 000-1_get_tag-ent.py &
input:  ../input/train.csv.zip
        ../input/test.csv.zip
output: ../input/mk/train_tag.p #4
        ../input/mk/test_tag.p
        ../input/mk/train_ent.p #5
        ../input/mk/test_ent.p
        ../input/mk/train_chunks.p #6
        ../input/mk/test_chunks.p



nohup python -u 000-2_freq_entities.py &
input:  ../input/mk/train_stop-stem.csv.gz #3
        ../input/mk/test_stop-stem.csv.gz
        ../input/mk/train_ent.p #5
        ../input/mk/test_ent.p
output: ../nlp_source/word_freq.csv.gz
        ../nlp_source/ent_ORG_freq.csv.gz
        ../nlp_source/ent_GPE_freq.csv.gz
        ../nlp_source/ent_PERSON_freq.csv.gz
        ../nlp_source/ent_LOC_freq.csv.gz
        ../nlp_source/ent_EVENT_freq.csv.gz




nohup python -u 000-3_freq_removed_stop_words.py &
input:  ../input/mk/train_stop.csv.gz #2
        ../input/mk/test_stop.csv.gz
output: ../nlp_source/tf.csv.gz


nohup python -u 000-4_remove_stops_from_tag.py &
input:  ../input/mk/train_tag.p #4
        ../input/mk/test_tag.p
output: ../input/mk/train_tag-stop.p #7
        ../input/mk/test_tag-stop.p


nohup python -u 000-5_JNV.py &
input:  ../input/mk/train_tag.p #4
        ../input/mk/test_tag.p
output: ../input/mk/train_JNV.p
        ../input/mk/test_JNV.p

nohup python -u 000-6_w2v-d2v.py &


nohup python -u 000-X_GloVe-w2v.py &
nohup python -u 000-X_verb.py &


#==============================================================================
# 
#==============================================================================
nohup python -u 001_perfect_0123.py &
input:  ../input/mk/train_prep.csv.gz      #0
        ../input/mk/test_prep.csv.gz
        ../input/mk/train_stem.csv.gz      #1
        ../input/mk/test_stem.csv.gz
        ../input/mk/train_stop.csv.gz      #2
        ../input/mk/test_stop.csv.gz
        ../input/mk/train_stop-stem.csv.gz #3
        ../input/mk/test_stop-stem.csv.gz
output: 


nohup python -u 002_tag_04.py &
input:  ../input/mk/train_prep.csv.gz #0
        ../input/mk/test_prep.csv.gz
        ../input/mk/train_tag.p       #4
        ../input/mk/test_tag.p
output: 




nohup python -u 003_emd_0123.py &
input:  ../input/mk/train_prep.csv.gz      #0
        ../input/mk/test_prep.csv.gz
        ../input/mk/train_stem.csv.gz      #1
        ../input/mk/test_stem.csv.gz
        ../input/mk/train_stop.csv.gz      #2
        ../input/mk/test_stop.csv.gz
        ../input/mk/train_stop-stem.csv.gz #3
        ../input/mk/test_stop-stem.csv.gz
output: 




nohup python -u 004_how_0.py &
input:  ../input/mk/train_prep.csv.gz #0
        ../input/mk/test_prep.csv.gz
output: 




nohup python -u 005_difference_0.py &
input:  ../input/mk/train_prep.csv.gz #0
        ../input/mk/test_prep.csv.gz
output: 



nohup python -u 006_verb_4.py &
input:  ../input/mk/train_tag.p #4
        ../input/mk/test_tag.p
output: 



nohup python -u 007_best_0.py &
input:  ../input/mk/train_prep.csv.gz #0
        ../input/mk/test_prep.csv.gz
output: 



nohup python -u 008_condition_0.py &
input:  ../input/mk/train_prep.csv.gz #0
        ../input/mk/test_prep.csv.gz
output: 




nohup python -u 009_num_large_04.py &
input:  ../input/mk/train_prep.csv.gz #0
        ../input/mk/test_prep.csv.gz
        ../input/mk/train_tag.p       #4
        ../input/mk/test_tag.p
output: 



nohup python -u 010_bow_3.py &
input:  ../input/mk/train_stop-stem.csv.gz #3
        ../input/mk/test_stop-stem.csv.gz
        ../nlp_source/word_freq.csv.gz
output: 



nohup python -u 011_set_diff_7.py &
input:  ../input/mk/train_tag-stop.p #7
        ../input/mk/test_tag-stop.p
output: 



nohup python -u 012_firstNVO_similarity_04.py &
input:  ../input/mk/train_prep.csv.gz #0
        ../input/mk/test_prep.csv.gz
        ../input/mk/train_tag.p       #4
        ../input/mk/test_tag.p
output: 



nohup python -u 013_sentence_similarity_2.py &
input:  
        
output: 



nohup python -u 014_stop-vector.py &
input:  
        
output: 



nohup python -u 015_1intersection.py &
input:  
        
output: 



nohup python -u 100_from_abish.py &
input:  ../input/mk/train_prep.csv.gz #0
        ../input/mk/test_prep.csv.gz
        ../input/mk/train_stem.csv.gz #1
        ../input/mk/test_stem.csv.gz
        ../input/mk/train_stop.csv.gz #2
        ../input/mk/test_stop.csv.gz
        ../input/mk/train_stop-stem.csv.gz #3
        ../input/mk/test_stop-stem.csv.gz
        
output: 



nohup python -u 101_from_anokas.py &
input:  ../input/mk/train_prep.csv.gz #0
        ../input/mk/test_prep.csv.gz
        ../input/mk/train_stem.csv.gz #1
        ../input/mk/test_stem.csv.gz
        ../input/mk/train_stop.csv.gz #2
        ../input/mk/test_stop.csv.gz
        ../input/mk/train_stop-stem.csv.gz #3
        ../input/mk/test_stop-stem.csv.gz
        
output: 



nohup python -u 102_from_the1owl.py &
input:  ../input/mk/train_prep.csv.gz #0
        ../input/mk/test_prep.csv.gz
        ../input/mk/train_stem.csv.gz #1
        ../input/mk/test_stem.csv.gz
        ../input/mk/train_stop.csv.gz #2
        ../input/mk/test_stop.csv.gz
        ../input/mk/train_stop-stem.csv.gz #3
        ../input/mk/test_stop-stem.csv.gz
        
output: 



nohup python -u 103-0_from_qqgeogor.py &
input:  ../input/mk/train_prep.csv.gz #0
        ../input/mk/test_prep.csv.gz
        ../input/mk/train_stem.csv.gz #1
        ../input/mk/test_stem.csv.gz
        ../input/mk/train_stop.csv.gz #2
        ../input/mk/test_stop.csv.gz
        ../input/mk/train_stop-stem.csv.gz #3
        ../input/mk/test_stop-stem.csv.gz
        
output: 



nohup python -u 990_run-all.py &



nohup python -u 998-1_cv.py &


nohup python -u 998-2_cv_all.py &

nohup python -u 999_down.py &
"""

from glob import glob

files = [f for f in sorted(glob('*.py')) if f[0].isdigit()]

for f in files:
    print("nohup python -u {} &".format(f))

print("""#==============================================================================
# SUCCESS !!! {}
#==============================================================================
""".format(__file__))

