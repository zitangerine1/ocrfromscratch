from CRNN import *
import keras.backend as K

CRNNmodel.load_weights('/home/couch/Documents/GitHub/ocrfromscratch-WIP/model/bestmodel.h5')
prediction = CRNNmodel.predict(valid_img[10:20])

out = K.get_value(K.ctc_decode(prediction, input_length = np.ones(prediction.shape[0]) * prediction.shape[1], greedy = True)[0][0])

i = 10

for x in out:
    print("original_text =  ", valid_orig_txt[i])
    print("predicted text = ", end = '')
    
    for p in x:
        if int(p) != -1:
            print(charlist[int(p)], end = '')
            
    print('\n')
    i += 1