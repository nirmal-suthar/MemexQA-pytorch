class Dataset():
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