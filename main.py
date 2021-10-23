from fastapi import FastAPI
from pydantic import BaseModel
import ssl

from keras.models import load_model
import pickle
from genderprediction import *
from namedetection import *

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
# Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
# Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


'''
Load the model
'''

'''   Name Detection Model  '''

noname_freqlist_path = './name_detection/noname_freqlist.pickle'
name_freqlist_path   = './name_detection/name_freqlist.pickle'

#Open the noname_frequence list
with open(noname_freqlist_path, 'rb') as handle:
    noname_freqlist = pickle.load(handle)
    handle.close()
    
#Open the name_fequence list
with open(name_freqlist_path, 'rb') as handle:
    name_freqlist = pickle.load(handle)
    handle.close()




'''   Gender Prediciton Model  '''


CHARLEVEL_MODEL_PATH      =   './gender_prediction/genderpred_lstmsequence.h5'
CHARLEVEL_TOKENIZER       =   './gender_prediction/genderpred_lstmsequence_tokenizer.pickle'

#Load gender prediction model
model_genderpred         = load_model(CHARLEVEL_MODEL_PATH)

#Load tokenizer
with open(CHARLEVEL_TOKENIZER, 'rb') as handle:
    gender_pred_tokenizer = pickle.load(handle)
    handle.close()





'''
Create a post request body
'''
class Body(BaseModel):
    name: str


'''

Deploy the app

'''

    
    



app = FastAPI()

@app.post("/get_gender")
def returnGender(body: Body):

    #Get the post params
    input_name = body.name

    #Name detection
    name_status = detect_name(input_name, noname_freqlist, name_freqlist)

    if name_status == 'Name':
        name = prepare_X([input_name], gender_pred_tokenizer)
        gender = model_genderpred.predict(name, verbose=1)
        if np.argmax(gender) == 1:
            return 'Female'
        return 'Male'
        
    return 'undefined'



