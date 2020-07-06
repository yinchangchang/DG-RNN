# Data format

# ehr_vocab.json
-	A list of codes

# train.json
-	A list of patient data.
-	Each patient's data are represented as [dict, 0/1].
	-	dict:
		-	key: time of a visit
		-	value: a list the codes in the visit

# valid.json
The same format as train.json

# test.json
The same format as train.json
