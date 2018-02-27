#/use/bin/python3.6

#INPUT :  get_data(clf_type, min_class_size) -> minimum classification type and min class size as input
#OUTPUT : X, y, max_len, nb_class -> X and y for ML. max_len of domain, number of classes.


import pandas as pd
import string
from keras.preprocessing import sequence
import pickle
import os 

#cd ~/Documents/austin/dga_archive2

def get_dga_class_info():
    import pickle
    dga_df = pd.read_csv('./data/dgadata.csv',header=None)
    
    info=[]
    domains = set(list(dga_df[1]))
    for domain in domains:
        info.append(  (len(dga_df.loc[dga_df[1] == domain]), domain))
    with open("./data/data_info.txt", "wb") as fp:       #Pickling
        pickle.dump(info, fp)
    
    print(info)
    print("Information saved at './data/data_info.txt ")
    
    return info


def get_data_min_class(info, clf, min_class_size):

    dga_df = pd.read_csv('./data/dgadata.csv',header=None)
    count=0
    min_class_domains=[]
    for x in info :
        if x[0] > min_class_size:
            min_class_domains.append(x[1])
            count=count+1
#            
#    with open("./data/info_"+ str(min_class_size) +"_class.txt", "wb") as fp:   
#        pickle.dump(min_class_domains, fp)
#            
    dga = (dga_df.loc[ dga_df[1] == min_class_domains[0] ]).sample(n=min_class_size) #initial	
    for x in min_class_domains[1:]:
	    dga= dga.append(( dga_df.loc[ dga_df[1] == x ]).sample(n=min_class_size) )
    
    del(dga_df)
    
    if clf == 1:
        dga[1]='dga'
    
    alexa_df= pd.read_csv('./data/top-1m.csv',header=None)
    alexa_df[0] = alexa_df[1] # Its index in alexa_df[0]
    alexa_df[1]='alexa'
    alexa = (alexa_df).sample(n=min_class_size)
            
    del(alexa_df)
    
    frames = [dga, alexa]
    df = pd.concat(frames)
    
    return data_ml(df)

def data_ml(df):
    
	nb_class= len(set(df.iloc[:,1]))
	print("Number of classes : ", nb_class)

	trainX=[]
	trainY=[]
	
	max_len=0
	for domain in df.iloc[:,1]:
	    if len(domain)> max_len:
	        max_len= len(domain)
	print("Maximum domain Length : ",max_len)
	        
	for index,row in df.iterrows():
	    trainX.append(row[0])
	    trainY.append(row[1])
	
	from sklearn.preprocessing import LabelBinarizer
	encoder = LabelBinarizer()
	trainY = encoder.fit_transform(trainY)

	#Generating list of all valid URL characters
	all_chars= string.ascii_lowercase + string.ascii_uppercase + string.digits + "-_."
	valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(all_chars)))}
	
	domains = [[valid_chars[y] for y in x] for x in trainX]
	domains = sequence.pad_sequences(domains, maxlen=max_len)
	
	X = sequence.pad_sequences(domains, maxlen=max_len)
	y = sequence.pad_sequences(trainY, maxlen=nb_class)
	
	df = {'X': X, 'y' :y,'max_len' : max_len,'nb_class' : nb_class}
	return  {X,y, max_len, nb_class}


def get_data(clf,min_class_size=100000):

    import os.path

    if not os.path.isfile("./data/data_"+ str(min_class_size) +".csv"):
        # Saving class info
        if not os.path.isfile('./data/data_info.txt'): 
            print('Info of file creating')
            info = get_dga_class_info()     
        else:
            with open("./data/data_info.txt", "rb") as fp:
                info = pickle.load(fp)

        print("Creating data with min size : ",min_class_size)
        df=get_data_min_class(info,clf, min_class_size)
        
        #Loading saved min_data_size
        
        #p1 = Process(target=get_dga_class_info, args=(dga_df,) )
        #info = p1.start()
        #p2 = Process(target=get_dga_min_class, args=(dga_df, info, min_data_size,))
        #dga = p2.start()
        
        	# saving data
        os.system('say "Wanna save the data ?"')
        if(input('Do you wanna save the df (Alexa + DGA) ? (y/n) : ') == 'y'):
            df.to_csv('./data/data_'+str(min_class_size)+'.csv',columns=None)
            print("Data saved at ./data/data_"+str(min_class_size)+'.csv' )
    else:
        print("Loading Saved data")
        df = pd.read_csv('./data/data_'+ str(min_class_size) +".csv",index_col=None)
    
    print("Processing Data for Machine leaning ")
    
    return df
