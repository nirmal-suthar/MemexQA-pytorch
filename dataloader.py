import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np
from collections import Counter
import itertools
from itertools import takewhile

class arg():
    def __init__(self):
        self.data_path = '../prepro_new/train_data.p'
        self.shared_path = '../prepro_new/train_shared.p'
        self.batch_size = 16
        self.data_workers = 4
        
config = arg()

def get_loader(data_path, shared_path, train=False, val=False, test=False):
    """ Returns a data loader for the desired split """
    assert train + val + test == 1, 'need to set exactly one of {train, val, test} to True'
    
    with open(data_path,"rb")as f:
        data = pickle.load(f)
    with open(shared_path,"rb") as f:
        shared = pickle.load(f,encoding='latin1') # this will be added later with word id, either new or load from exists
    
    split = memexQA(data, shared)
    loader = DataLoader(
        split,
        batch_size=config.batch_size,
        shuffle=train,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
    )
    return loader


class memexQA(Dataset):
    # data should be 
    """
    data = {
        'q':q,
        'cq':cq,
        'y':y,
        'cy':cy,
        'aid':aid, # each is a list of aids
        'qid':qid,
        'idxs':idxs,
        'cs':cs, # each is a list of wrong choices
        'ccs':ccs,
        ################# new for a mini batch##################
        album_title = []
        album_title_c = []
        album_description = []
        album_description_c = []
        where = []
        where_c = []
        when = []
        when_c = []
        photo_titles = []
        photo_titles_c = []
        photo_ids = [] -> original pids , string
        photo_idxs = [] -> pids transform to the image_feat_matrix idx

        image_feat_matrix
    }
    in mini batch,
    data added all other things it need from the shared
    , shared is the whole shared dict

    """
    
    def __init__(self, data, shared):
        


        self.data = data # data is a dict {"q":[],"cq"}
        self.shared = shared

        self.token_to_index = self.shared['vocab']

        self.valid_idxs = range(self.get_data_size())
        self.num_examples = len(self.valid_idxs)

        #encoded data
        self.e_q = [self._encode_title(q) for q in self.data['q']]
        self.e_y = [self._encode_title(y) for y in self.data['y']]
        self.e_cs = [[self._encode_title(cs_w) for cs_w in cs_single ] for cs_single in self.data['cs']]

        
        self.month2idx = {
            'january':1,
            'february':2,
            'march':3,
            'april':4,
            'may':5,
            'june':6,
            'july':7,
            'august':8,
            'september':9,
            'october':10,
            'november':11,
            'december':12
        }
        
    @property
    def max_title_length(self):
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.data['q']))
        return self._max_length
    
    def _encode_title(self, title):
        """ Turn a title into a vector of indices and a title length """
        vec = torch.zeros(self.max_title_length).long()
        for i, token in enumerate(title):
            index = self.token_to_index.get(token, 0)
            vec[i] = index
        return vec, len(title)
    
    def _encode_when(self, title):
        
        vec = torch.zeros(3).long()
        vec[0]=self.month2idx[title[1]]
        vec[1]=int(title[2])
        vec[2]=int(title[3])
        return vec
    
    def get_data_size(self):
        return len(next(iter(self.data.values()))) # get one var "q" and get the len
    
    def __getitem__(self, idx):
        
        q = self.e_q[idx]
        y = self.e_y[idx]
        cs = self.e_cs[idx]
        aid = self.data['aid'][idx]

        # one shared album info for one qa, could be multiple albums
        album_title = []
        album_title_c = []
        album_description = []
        album_description_c = []
        album_where = []
        album_where_c = []
        album_when = []
        album_when_c = []
        photo_titles = []
        photo_titles_c = []
        album_pid = []


        for albumId in aid:
            album = self.shared['albums'][albumId]
            album_title.append(self._encode_title(album['title']))
            album_title_c.append(album['title_c'])
            album_description.append(self._encode_title(album['description']))
            album_description_c.append(album['description_c'])
            album_where.append(self._encode_title(album['where']))
            album_when.append(self._encode_when(album['when']))
            album_where_c.append(album['where_c'])
            album_when_c.append(album['when_c'])
            photo_titles.append([self._encode_title(title) for title in album['photo_titles']])
            photo_titles_c.append(album['photo_titles_c'])
            album_pid.append(album['photo_ids'])
            
        
        pid2feat = {}
        
        for pid_collection in album_pid:
            for pid in pid_collection:
                if not pid in pid2feat:
                    pid2feat[pid] = self.shared['pid2feat'][pid]
                              
        return q, y, cs, aid, album_title, album_description, album_where, album_when, photo_titles, album_pid, pid2feat 
    
    
    def __len__(self):
        return self.num_examples

get_loader(config.data_path, config.shared_path, train=True)