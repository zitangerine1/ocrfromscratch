import numpy as np
from SegmentPage import segment_to_line
from SegmentLine import segment_to_words
from RecogniseWord import recognize_words

line_img_array = segment_to_line('/home/couch/Documents/GitHub/ocrfromscratch-WIP/livetest/1_PECLqzgGmIIMkjHX_Y9GqQ.webp')

full_index_indicator = []
all_words_list = []
len_line_arr = 0

for index, im in enumerate(line_img_array):
    line_indicator, word_array = segment_to_words(im, index)
    
    for k in range(len(word_array)):
        full_index_indicator.append(line_indicator[k])
        all_words_list.append(word_array[k])
    
    len_line_arr += 1
    
all_words_list = np.array(all_words_list)
recognize_words(full_index_indicator, all_words_list, len_line_arr)