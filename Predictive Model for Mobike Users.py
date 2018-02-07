"""
Predictive Model for Mobike Users
Co-authors: xingyu Fu && shen Zheng
Institutions: Sun Yat_sen University && JiNan University && Shining Midas Private Fund
For contact: 443518347@qq.com
All Rights Reserved
"""
"""Import Libraries"""
from geohash import decode_exactly
from pandas import read_csv
from sklearn.neighbors import KDTree
from scipy.optimize import minimize
import numpy as np
import math
import csv
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import grid_search 
from random import seed 
from random import randrange
import os
from collections import OrderedDict


"""Change the Working Directory"""
dire = "C:/Users/fxy/Documents/Academic/AI/Fintechlab/Code/Mobike_Contest"
os.chdir(dire)
del dire


"""Some Constants"""
N=3 #Number of predictions we are going to make
MAX_KM=3.0 #The Maximal distance we allow for a single bike trip 
           #(The average is 0.815 ; 1000000th is 0.493 ; 2000000th is 0.799 ; 3000000th is 1.67 ; 3100000th is 2.16 ; 3170000th is 3.212)
KD=5 #The number of destinations KDTree predict  
TRICK_NUM=2 #The number of destinations Trick predict  
Neighbor_Condition=0.5 #The distance threshold to characterize neighbors   
Method4_Top = 5               # If the above threshold is exceeded , then we pick up 5 destinations to make prediction      
Bayes_Num=4 #The maximal number of Negative examples we allow the algorithm to predict
stdv = 1 #The Standard Variance which the Gaussian PDF uses
time_scaler1=12.0
lat_scaler1=0.01
lon_scaler1=0.01

time_scaler2=1.20
lat_scaler2=0.1
lon_scaler2=0.1

Expand_Num = 4 #For each negative example , how many neighbours should a single Expander find.


"""Construct A Dictionary whose key is the orderid and the corresponding value is the specific date(TEST DATASET)"""
#Useful when we implement the Method6 in "test" mode
file_name_test = "test.csv"
orderid_time_test = read_csv(file_name_test,usecols=[0,4])
orderid_time_test = orderid_time_test.values
for row in orderid_time_test:
    row[1] = int( row[1][8:10] )
orderid_time_test = dict( orderid_time_test )
del file_name_test 

file_name_train="train.csv"
order_time_train=read_csv(file_name_train,usecols=[0,4])
order_time_train=order_time_train.values
for row in order_time_train:
    row[1] = int( row[1][8:10] )
order_time_train= dict(order_time_train)
del file_name_train

orderid_time_test.update(order_time_train)
del order_time_train


"""Transform the date into Weekday(0) and Weekend(1)"""
def work_play(inf):
    #prepare data
    date=int( inf[8:10] )
    #handle date
    set_end={13,14,20,21,27,28}
    dateflag=int(not (date in set_end))
    return dateflag  


"""Transform the time information into the float formation of hour (e.g. 13:27->13.45)"""
def hour_minute(inf):
    hour=float(inf[11:13])
    minute=float(inf[14:16])/60.0
    return hour+minute


"""Calculate the distance between two points given their Latitude and Longitude (KM)"""
def distan(lat1 ,lng1 ,lat2 ,lng2 ):
    radlat1=math.radians(lat1)  
    radlat2=math.radians(lat2)  
    a=radlat1-radlat2  
    b=math.radians(lng1)-math.radians(lng2)  
    s=2*math.asin(math.sqrt(math.pow(math.sin(a/2),2)+math.cos(radlat1)*math.cos(radlat2)*math.pow(math.sin(b/2),2)))  
    earth_radius=6378.137  
    s=s*earth_radius  
    if s<0:  
        return -s  
    else:  
        return s


"""Gaussian Distribution PDF (For Bayes)"""
def Gaussian_PDF(x,u,s):
    exponential=math.exp( -( ((x-u)**2)/(2*s**2) ) )
    return ( 1.0/(math.sqrt(2*math.pi)*s) ) * exponential


"""Bulid dictionary"""
def Build_dict(Set,Tag,Way):
    #Set indicates the dataset which the dictionary is based on
    #Tag indicates the key of the dictionary
    #Way indicates the method that we manipulate the dataset
    result=dict()
    if Way==0 :#Add to the same list if tag repeated
        for record in Set:
            if record[Tag] in result:
                result[ record[Tag] ].append(record)
            else:
                result[ record[Tag] ]=[]
                result[ record[Tag] ].append(record)
    else :#Count frequency if tag repeated (way==1)
        for record in Set:
            if record[Tag] in result:
                result[ record[Tag] ]+=1
            else:
                result[ record[Tag] ]=1
    return result


def random_subset(X,Y, split=0.01): 
    X_sub =list()
    Y_sub =list() 
    sub_size = split * len(X)
    while len(X_sub) < sub_size: 
        index = randrange( len(X) ) 
        X_sub.append( X[index] )
        Y_sub.append( Y[index] )
    return X_sub,Y_sub

    
"""Feature Vector Generator"""
def Feature_Vector(methods, real_record, prediction_record, prediction_destination, Knowledges):
    """The Features constructed only by the real_record"""
    F0 = 1 if (real_record[0] in Knowledges[0]) else 0 #Old Clients(1) or New Clients(0)
    F1 = real_record[1] # Wether or not the start point is weekend
    F2 = real_record[2] # The specific time in the start point
    F3 = real_record[4] # The latitude of the start point
    F4 = real_record[5] # The longitude of the start point
    """The Features constructed by the methods you use to create this specific negative examples"""
    F5 = 1 if (methods == 1) else 0 #Use Method1
    F6 = 1 if (methods == 2) else 0 #Use Method2
    F7 = 1 if (methods == 3) else 0 #Use Method3
    F8 = 1 if (methods == 4) else 0 #Use Method4
    F9 = 1 if (methods == 5) else 0 #Use Method5
    F10= 1 if (methods == 6) else 0 #Use Method6
    FBayes = 1 if (methods == 7) else 0 #Use Method7
    FtimtKD = 1 if (methods == 8) else 0 #Use Method8
    """The Features constructed by the personal history records of this specific user"""
    F11 = 0
    F12 = 0
    F13 = 0
    F14 = 0
    F15 = 0
    F16 = 0
    if F0 != 0:
        personal_history = Knowledges[0][ real_record[0] ] #From "train" (operator)
        bool_same_start=[(1 if (rec[3]==real_record[3]) else 0) for rec in personal_history]
        F11= sum( bool_same_start ) #How many times the specific user start his trip in this given start
        F12= float(F11)/float( len( bool_same_start ) ) #The frequency of this given start point in the user's history
        bool_same_end=[ (1 if rec[6]==prediction_destination else 0) for rec in personal_history ]
        F13= sum( bool_same_end ) #How many times the specific user go to this prediction_destination
        F14= float(F13)/float( len(bool_same_end) ) #The frequency of this given destination in the user's history
        bool_same_start_end= [(1 if (bool_same_start[i] and bool_same_end[i]) else 0) for i in range( len(personal_history) )] 
        F15= sum( bool_same_start_end ) #How many times the specific route has been biked by this user
        F16= float(F15)/float( len(bool_same_start_end) ) #The frequency of F15
        del bool_same_start
        del bool_same_end
        del bool_same_start_end
        del personal_history
    """The Features constructed by Popularity-Map"""
    Popularity_Map_depart=Knowledges[3]
    Popularity_Map_destin=Knowledges[4]
    F17 = 0
    F18 = 0
    F19 = 0
    if real_record[3] in Popularity_Map_depart:
        F17= Popularity_Map_depart[ real_record[3] ] #Popularity of this given start in the city
    if prediction_destination in Popularity_Map_destin:
        F18= Popularity_Map_destin[ prediction_destination ] #Popularity of this given destination in the city
    if real_record[3] in Knowledges[2]:
        Same_start_users_records=Knowledges[2][ real_record[3] ] # From operator ("train")
        F19= sum( [ (1 if rec[6]==prediction_destination else 0) for rec in Same_start_users_records ] ) #Popularity of this given path in the city
        del Same_start_users_records
    del Popularity_Map_depart
    del Popularity_Map_destin
    """The Features constructed by the prediction_record"""
    F20= prediction_record[1] #Wether or not the prediction_record is weekend
    F21= (1 if (F1==F20) else 0) #Wether or not F1 and F20 are the same
    """Coefficient of Universal Gravity"""
    F22 = 0 # Formula of Gravity -> (GMm)/r^2
    if (methods == 2) or (methods == 6):
        r=distan(real_record[4],real_record[5],prediction_record[4],prediction_record[5])
        if r!=0:
            F22= float(F17*F18)/(r**2)
        else:
            F22= np.nan
    else:
        r=distan(real_record[4],real_record[5],prediction_record[7],prediction_record[8])
        if r!=0:
            F22= float(F17*F18)/(r**2)
        else:
            F22= np.nan
    # The time difference between the real_record and prediction_record
    F23=0
    if (methods == 2) or (methods == 6):
        F23=np.nan
    else:
        F23=-(math.fabs(math.fabs(real_record[2]-prediction_record[2])-12)-12)
        
    # The multiplication between start_point popularity and destination popularity
    F24 = F17*F18
    
    #Slicing the time into several sections (USing One Hot Encoding)
    F25 = 1 if(real_record[2]>=6.0 and real_record[2]<9.5) else 0
    F26 = 1 if(real_record[2]>=9.5 and real_record[2]<11.5) else 0
    F27 = 1 if(real_record[2]>=11.5 and real_record[2]<14) else 0
    F28 = 1 if(real_record[2]>=14.0 and real_record[2]<17.0) else 0
    F29 = 1 if(real_record[2]>=17.0 and real_record[2]<20.0) else 0
    F30 = 1 if(real_record[2]>=20.0 and real_record[2]<23.0) else 0
    F31 = 1 if(real_record[2]>=23.0 or real_record[2]<6.0) else 0
    
    #Feature by double_dict
    dictionary=dict()
    if F25 == 1:
        dictionary = Knowledges[6][1]
    elif F26 == 1:
        dictionary = Knowledges[6][2]
    elif F27 == 1:
        dictionary = Knowledges[6][3]
    elif F28 == 1:
        dictionary = Knowledges[6][4]
    elif F29 == 1:
        dictionary = Knowledges[6][5]
    elif F30 == 1:
        dictionary = Knowledges[6][6]
    else: #F31 == 1
        dictionary = Knowledges[6][7]
    
    F32 = 0
    if prediction_destination in dictionary:
        F32 = float(dictionary[ prediction_destination ]) / float(dictionary[ "sum" ])
    
    if (methods != 6) and (methods !=2) :
        Vector=[ F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,FBayes,FtimtKD,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,F21,F22,F23,F24,F25,F26,F27,F28,F29,F30,F31,F32,prediction_destination,prediction_record[7],prediction_record[8] ]
    else:
        Vector=[ F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,FBayes,FtimtKD,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,F21,F22,F23,F24,F25,F26,F27,F28,F29,F30,F31,F32,prediction_destination,prediction_record[4],prediction_record[5] ]
    return Vector
       

def Feature_Vector1(methods, real_record, prediction_record, neighbor_record, prediction_destination, Knowledges):
    """The Features constructed only by the real_record"""
    F0 = 1 if (real_record[0] in Knowledges[0]) else 0 #Old Clients(1) or New Clients(0)
    F1 = real_record[1] # Wether or not the start point is weekend
    F2 = real_record[2] # The specific time in the start point
    F3 = real_record[4] # The latitude of the start point
    F4 = real_record[5] # The longitude of the start point
    """The Features constructed by the methods you use to create this specific negative examples"""
    F5 = 1 if (methods == 1) else 0 #Use Method1
    F6 = 1 if (methods == 2) else 0 #Use Method2
    F7 = 1 if (methods == 3) else 0 #Use Method3
    F8 = 1 if (methods == 4) else 0 #Use Method4
    F9 = 1 if (methods == 5) else 0 #Use Method5
    F10= 1 if (methods == 6) else 0 #Use Method6
    FBayes = 1 if (methods == 7) else 0 #Use Method7
    FtimtKD = 1 if (methods == 8) else 0 #Use Method8
    """The Features constructed by the personal history records of this specific user"""
    F11 = 0
    F12 = 0
    F13 = 0
    F14 = 0
    F15 = 0
    F16 = 0
    if F0 != 0:
        personal_history = Knowledges[0][ real_record[0] ] #From "train" (operator)
        bool_same_start=[(1 if (rec[3]==real_record[3]) else 0) for rec in personal_history]
        F11= sum( bool_same_start ) #How many times the specific user start his trip in this given start
        F12= float(F11)/float( len( bool_same_start ) ) #The frequency of this given start point in the user's history
        bool_same_end=[ (1 if rec[6]==prediction_destination else 0) for rec in personal_history ]
        F13= sum( bool_same_end ) #How many times the specific user go to this prediction_destination
        F14= float(F13)/float( len(bool_same_end) ) #The frequency of this given destination in the user's history
        bool_same_start_end= [(1 if (bool_same_start[i] and bool_same_end[i]) else 0) for i in range( len(personal_history) )] 
        F15= sum( bool_same_start_end ) #How many times the specific route has been biked by this user
        F16= float(F15)/float( len(bool_same_start_end) ) #The frequency of F15
        del bool_same_start
        del bool_same_end
        del bool_same_start_end
        del personal_history
    """The Features constructed by Popularity-Map"""
    Popularity_Map_depart=Knowledges[3]
    Popularity_Map_destin=Knowledges[4]
    F17 = 0
    F18 = 0
    F19 = 0
    if real_record[3] in Popularity_Map_depart:
        F17= Popularity_Map_depart[ real_record[3] ] #Popularity of this given start in the city
    if prediction_destination in Popularity_Map_destin:
        F18= Popularity_Map_destin[ prediction_destination ] #Popularity of this given destination in the city
    if real_record[3] in Knowledges[2]:
        Same_start_users_records=Knowledges[2][ real_record[3] ] # From operator ("train")
        F19= sum( [ (1 if rec[6]==prediction_destination else 0) for rec in Same_start_users_records ] ) #Popularity of this given path in the city
        del Same_start_users_records
    del Popularity_Map_depart
    del Popularity_Map_destin
    """The Features constructed by the prediction_record"""
    F20= prediction_record[1] #Wether or not the prediction_record is weekend
    F21= (1 if (F1==F20) else 0) #Wether or not F1 and F20 are the same
    """Coefficient of Universal Gravity"""
    F22 = 0 # Formula of Gravity -> (GMm)/r^2
    r=distan(real_record[4],real_record[5],neighbor_record[7],neighbor_record[8])
    if r!=0:
        F22= float(F17*F18)/(r**2)
    else:
        F22= np.nan
    # The time difference between the real_record and prediction_record
    F23=0
    if (methods == 2) or (methods == 6):
        F23=np.nan
    else:
        F23=-(math.fabs(math.fabs(real_record[2]-prediction_record[2])-12)-12)
        
    # The multiplication between start_point popularity and destination popularity
    F24 = F17*F18
    
    #Slicing the time into several sections (USing One Hot Encoding)
    F25 = 1 if(real_record[2]>=6.0 and real_record[2]<9.5) else 0
    F26 = 1 if(real_record[2]>=9.5 and real_record[2]<11.5) else 0
    F27 = 1 if(real_record[2]>=11.5 and real_record[2]<14) else 0
    F28 = 1 if(real_record[2]>=14.0 and real_record[2]<17.0) else 0
    F29 = 1 if(real_record[2]>=17.0 and real_record[2]<20.0) else 0
    F30 = 1 if(real_record[2]>=20.0 and real_record[2]<23.0) else 0
    F31 = 1 if(real_record[2]>=23.0 or real_record[2]<6.0) else 0
    
    #Feature by double_dict
    dictionary=dict()
    if F25 == 1:
        dictionary = Knowledges[6][1]
    elif F26 == 1:
        dictionary = Knowledges[6][2]
    elif F27 == 1:
        dictionary = Knowledges[6][3]
    elif F28 == 1:
        dictionary = Knowledges[6][4]
    elif F29 == 1:
        dictionary = Knowledges[6][5]
    elif F30 == 1:
        dictionary = Knowledges[6][6]
    else: #F31 == 1
        dictionary = Knowledges[6][7]
    
    F32 = 0
    if prediction_destination in dictionary:
        F32 = float(dictionary[ prediction_destination ]) / float(dictionary[ "sum" ])
    
    Vector=[ F0,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,FBayes,FtimtKD,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,F21,F22,F23,F24,F25,F26,F27,F28,F29,F30,F31,F32,prediction_destination,neighbor_record[7],neighbor_record[8] ]
    
    return Vector

 
    
"""The destination to which the user used to go(we do not fix the start)"""
def Method1(real_record,negative_examples,Knowledges,operator):
    if real_record[0] not in Knowledges[0]:
        return None
    personal_history = Knowledges[0][ real_record[0] ]#originated from operator (Must belongs to train)
    for prediction_record in personal_history:
        if distan(real_record[4],real_record[5],prediction_record[7],prediction_record[8]) > MAX_KM:
            # If the distance between the start and destination is too large, then we say that it is impossible for them to form a bike trip.
            continue 
        else:
            prediction_destination=prediction_record[6]
            negative=Feature_Vector(1, real_record, prediction_record, prediction_destination, Knowledges)
            negative_examples.append(negative)
            #Search the neighbours of the predicted destination
            Expander = Knowledges[8]
            dist,ind = Expander.query( [ [prediction_record[7]/lat_scaler1 , prediction_record[8]/lon_scaler1] ], k=Expand_Num)
            for j in range( Expand_Num ):
                prediction_destination=operator[ ind[0][j] ][6]
                neighbor_record = operator[ ind[0][j] ]
                negative=Feature_Vector1(1, real_record, prediction_record, neighbor_record, prediction_destination, Knowledges)
                negative_examples.append(negative)
    return None


"""The start from which the user used to start(we do not fix the destination)"""
def Method2(real_record,negative_examples,Knowledges,operator) :
    if real_record[0] not in Knowledges[0]:
        return None
    personal_history=Knowledges[0][ real_record[0] ] #from operator(train)
    for prediction_record in personal_history: #prediction_record form operator while real_record from operated
        if distan(real_record[4],real_record[5],prediction_record[4],prediction_record[5]) > MAX_KM:
            # If the distance between the start and destination is too large, then we say that it is impossible for them to form a bike trip.
            continue 
        else:
            prediction_destination=prediction_record[3]
            negative=Feature_Vector(2, real_record, prediction_record, prediction_destination, Knowledges)
            negative_examples.append(negative)
            #Search the neighbours of the predicted destination
            Expander = Knowledges[8]
            dist,ind = Expander.query( [ [prediction_record[4]/lat_scaler1 , prediction_record[5]/lon_scaler1] ], k=Expand_Num)
            for j in range( Expand_Num ):
                prediction_destination=operator[ ind[0][j] ][6]
                neighbor_record = operator[ ind[0][j] ]
                negative=Feature_Vector1(2, real_record, prediction_record, neighbor_record, prediction_destination, Knowledges)
                negative_examples.append(negative)
    return None


"""The destination to which the user used to go in this given start (3 is in 1)"""
def Method3(real_record,negative_examples,Knowledges,operator):
    if real_record[0] not in Knowledges[0]:
        return None
    personal_history=Knowledges[0][ real_record[0] ] #from operator
    for prediction_record in personal_history:
        if real_record[3] != prediction_record[3]:
            #We are now looking for the personal records with the same start 
            continue
        else:
            prediction_destination=prediction_record[6]
            negative=Feature_Vector(3, real_record, prediction_record, prediction_destination, Knowledges)
            negative_examples.append(negative)
            #Search the neighbours of the predicted destination
            Expander = Knowledges[8]
            dist,ind = Expander.query( [ [prediction_record[7]/lat_scaler1 , prediction_record[8]/lon_scaler1] ], k=Expand_Num)
            for j in range( Expand_Num ):
                prediction_destination=operator[ ind[0][j] ][6]
                neighbor_record = operator[ ind[0][j] ]
                negative=Feature_Vector1(3, real_record, prediction_record, neighbor_record,prediction_destination, Knowledges)
                negative_examples.append(negative)
    return None    


"""The destination to which all the users used to go from this start"""
def Method4(real_record,negative_examples,Knowledges,operator):
    if real_record[3] not in Knowledges[2]:
        return None
    destinations_from_record = Knowledges[2][ real_record[3] ] #from operator(Must be in train)
    
    des_count=dict()
    des_record=dict()
    for record in destinations_from_record:
        if record[6] not in des_count:
            des_count[ record[6] ] = 1
            des_record[ record[6] ] = record
        else:
            des_count[ record[6] ] += 1
    
    des_count=sorted(des_count.items(),key=lambda d:d[1],reverse=False )
    content=0
    for loc in des_count:
        if content >= Method4_Top:
            break
        else:
            content+=1
            prediction_record=des_record[ loc[0] ]
            prediction_destination=loc[0]
            negative=Feature_Vector(4, real_record, prediction_record, prediction_destination, Knowledges)
            negative_examples.append(negative)
            #Search the neighbours of the predicted destination
            Expander = Knowledges[8]
            dist,ind = Expander.query( [ [prediction_record[7]/lat_scaler1 , prediction_record[8]/lon_scaler1] ], k=Expand_Num)
            for j in range( Expand_Num ):
                prediction_destination=operator[ ind[0][j] ][6]
                neighbor_record = operator[ ind[0][j] ]
                negative=Feature_Vector1(4, real_record, prediction_record, neighbor_record, prediction_destination, Knowledges)
                negative_examples.append(negative)
    return None
       

"""The prediction made by Collaborative Fliter(Based on KDTree)"""
"""Note: This KD Tree focus on location"""
def Method5(real_record,negative_examples,Knowledges,operator) :
    dist,ind=Knowledges[5].query([ [ (real_record[2])/time_scaler1, real_record[4]/lat_scaler1, real_record[5]/lon_scaler1] ] , k=KD)
    
    num=KD
    while len( set( [ operator[index][6]  for index in ind[0] ] ) ) <= N-1:
        num+=KD
        dist,ind=Knowledges[5].query([ [ (real_record[2])/time_scaler1, real_record[4]/lat_scaler1, real_record[5]/lon_scaler1] ] , k=num)
        
        
    for j in range(num):
        prediction_record=operator[ ind[0][j] ] #the neighbors of real_record are from "train"(operator) set
        prediction_destination=prediction_record[6]
        negative=Feature_Vector(5, real_record, prediction_record, prediction_destination, Knowledges)
        negative_examples.append(negative)
        #Search the neighbours of the predicted destination
        Expander = Knowledges[8]
        dist,inde = Expander.query( [ [prediction_record[7]/lat_scaler1 , prediction_record[8]/lon_scaler1] ], k=Expand_Num)
        for j in range( Expand_Num ):
            prediction_destination=operator[ inde[0][j] ][6]
            neighbor_record = operator[ inde[0][j] ]
            negative=Feature_Vector1(5, real_record, prediction_record, neighbor_record, prediction_destination, Knowledges)
            negative_examples.append(negative)
            
    return None
        

"""The prediction made by Round_Trip trick"""
def Method6(real_record,negative_examples,Knowledges,operator,mode):
    history_ed=Knowledges[1][ real_record[0] ]
    des_distance=dict()
    des_record=dict()
    for prediction_record in history_ed:
        if prediction_record[3] in des_distance:
            continue
        else:
            if (prediction_record[1] == real_record[1]) and ( distan(real_record[4],real_record[5],prediction_record[4],prediction_record[5])<=MAX_KM ) :
                des_distance[ prediction_record[3] ] = distan( real_record[4], real_record[5], prediction_record[4], prediction_record[5])
                des_record[ prediction_record[3] ] = prediction_record
            else:
                continue
    des_distance=sorted(des_distance.items(),key=lambda d:d[1],reverse=False )
    content=0
    for loc in des_distance[1:]:
        if content >= TRICK_NUM:
            break
        else:
            content+=1
            prediction_record=des_record[ loc[0] ]
            prediction_destination=loc[0]
            if orderid_time_test[ prediction_record[-1] ] == orderid_time_test[ real_record[-1] ]:
                negative=Feature_Vector(6, real_record, prediction_record, prediction_destination, Knowledges)
                negative_examples.append(negative)
                #Search the neighbours of the predicted destination
                Expander = Knowledges[8]
                dist,ind = Expander.query( [ [prediction_record[4]/lat_scaler1 , prediction_record[5]/lon_scaler1] ], k=Expand_Num)
                for j in range( Expand_Num ):
                    prediction_destination=operator[ ind[0][j] ][6]
                    neighbor_record = operator[ ind[0][j] ]
                    negative=Feature_Vector1(6, real_record, prediction_record, neighbor_record, prediction_destination, Knowledges)
                    negative_examples.append(negative)
            else:
                negative=Feature_Vector(2, real_record, prediction_record, prediction_destination, Knowledges)
                negative_examples.append(negative)
                #Search the neighbours of the predicted destination
                Expander = Knowledges[8]
                dist,ind = Expander.query( [ [prediction_record[4]/lat_scaler1 , prediction_record[5]/lon_scaler1] ], k=Expand_Num)
                for j in range( Expand_Num ):
                    prediction_destination=operator[ ind[0][j] ][6]
                    neighbor_record = operator[ ind[0][j] ]
                    negative=Feature_Vector1(2, real_record, prediction_record, neighbor_record, prediction_destination, Knowledges)
                    negative_examples.append(negative)
                
        
        
"""The prediction made by Naive Gaussian-Bayes Algorithm (Make Prediction Only Based On Time)(7 is in 1)"""
def Method7(real_record,negative_examples,Knowledges,operator):
    if real_record[0] not in Knowledges[0]: #If the current user is not old user , then we do not use Bayes
        return None
    personal_history = Knowledges[0][ real_record[0] ]#originated from operator (Must belongs to train)
    subhistory=[record for record in personal_history if record[1] == real_record[1]] # A subset of personal_history ,in which all the record share the same kind of date
    l=float( len( subhistory ) )
    time = real_record[2]
    if l != 0.0 : # If the subhistory is un-empty
        
        possible_destination=dict()
        possible_destination_record=dict()
        for record in subhistory:
            if record[6] not in possible_destination:
                possible_destination[ record[6] ] = 1.0
                possible_destination_record[ record[6] ] = record
            else:
                possible_destination[ record[6] ] +=1.0#Record the frequency of the destinations
        
        #Calculate the Bayes Probability of each possible destination and sort by probability
        for loc in possible_destination:
            def cost(u):
                return sum([(math.fabs(math.fabs(record[2]-u)-12)-12)**2 for record in subhistory if record[6]==loc])
            re=minimize( fun=cost,x0=(12.0,),method="SLSQP",bounds=((0,24),) ) 
            u=(re.x)[0]
            possible_destination[loc]=(possible_destination[loc]/l)*Gaussian_PDF(u+(math.fabs(math.fabs(time-u)-12)-12),u,stdv)
        possible_destination=sorted(possible_destination.items(),key=lambda d:d[1],reverse=True)
        if len(possible_destination) >= Bayes_Num:
            possible_destination = possible_destination[ 0 : Bayes_Num ]
        else:
            pass
        
        for loc in possible_destination:
            prediction_record = possible_destination_record[ loc[0] ] 
            prediction_destination = loc[0]
            negative=Feature_Vector(7, real_record, prediction_record, prediction_destination, Knowledges)
            negative_examples.append(negative)
            #Search the neighbours of the predicted destination
            Expander = Knowledges[8]
            dist,ind = Expander.query( [ [prediction_record[7]/lat_scaler1 , prediction_record[8]/lon_scaler1] ], k=Expand_Num)
            for j in range( Expand_Num ):
                prediction_destination=operator[ ind[0][j] ][6]
                neighbor_record = operator[ ind[0][j] ]
                negative=Feature_Vector1(7, real_record, prediction_record, neighbor_record, prediction_destination, Knowledges)
                negative_examples.append(negative)
            
        return None
    else:
        return None
      

"""The prediction made by Collaborative Fliter(Based on KDTree)"""
"""Note: This KD Tree focus on time"""
def Method8(real_record,negative_examples,Knowledges,operator) :
    dist,ind=Knowledges[7].query([ [ (real_record[2])/time_scaler2, real_record[4]/lat_scaler2, real_record[5]/lon_scaler2] ] , k=KD)
    
    num=KD
    while len( set( [ operator[index][6]  for index in ind[0] ] ) ) <= N-1:
        num+=KD
        dist,ind=Knowledges[7].query([ [ (real_record[2])/time_scaler2, real_record[4]/lat_scaler2, real_record[5]/lon_scaler2] ] , k=num)
        
        
    for j in range(num):
        prediction_record=operator[ ind[0][j] ] #the neighbors of real_record are from "train"(operator) set
        prediction_destination=prediction_record[6]
        negative=Feature_Vector(8, real_record, prediction_record, prediction_destination, Knowledges)
        negative_examples.append(negative)
        #Search the neighbours of the predicted destination
        Expander = Knowledges[8]
        dist,inde = Expander.query( [ [prediction_record[7]/lat_scaler1 , prediction_record[8]/lon_scaler1] ], k=Expand_Num)
        for j in range( Expand_Num ):
            prediction_destination=operator[ inde[0][j] ][6]
            neighbor_record = operator[ inde[0][j] ]
            negative=Feature_Vector1(8, real_record, prediction_record, neighbor_record, prediction_destination, Knowledges)
            negative_examples.append(negative)
    return None


"""Construct Negative Examples"""
def Negative_Examples(operated,operator,mode):
   X_train_xgboost=[]
   Y_train_xgboost=[]
   test_xgboost=OrderedDict()
   #Construct the name dictionary of all users in the operator("train") dataset
   #the value corresponding to each name is a list of personal records in the operator dataset
   namedict_or=Build_dict(operator,Tag=0,Way=0)
   
   #Construct the name dictionary of all users in the operated("test") dataset
   namedict_ed=Build_dict(operated,Tag=0,Way=0)
   
   #Construct the departure dictionary of all records in the operator dataset
   #the value corresponding to each departure point is a list of records which share the same departure point
   departdict_or=Build_dict(operator,Tag=3,Way=0)
   
   #Construct the popularity-map of all Departure points in the operator dataset
   Popularity_Map_departure=Build_dict(operator,Tag=3,Way=1)
   
   #Construct the popularity-map of all destinations in the operator dataset
   Popularity_Map_destination=Build_dict(operator,Tag=6,Way=1)
   
   #Construct KD Tree of the operator dataset
   #This KD Tree mainly focus on location (Little Concern About the Time)
   train_for_tree=[]
   for record in operator:
       sample_for_tree=[]
       sample_for_tree.append( record[2]/time_scaler1 )
       sample_for_tree.append( record[4]/lat_scaler1 )
       sample_for_tree.append( record[5]/lon_scaler1 )
       train_for_tree.append( sample_for_tree )
   tree1=KDTree( train_for_tree )
   
   #Construct KD Tree of the operator dataset
   #This KD Tree mainly focus on time (Little Concern About the Location)
   train_for_tree=[]
   for record in operator:
       sample_for_tree=[]
       sample_for_tree.append( record[2]/time_scaler2 )
       sample_for_tree.append( record[4]/lat_scaler2 )
       sample_for_tree.append( record[5]/lon_scaler2 )
       train_for_tree.append( sample_for_tree )
   tree2=KDTree( train_for_tree )
   
   #Construct KD Tree of the DESTINATION in operator dataset
   #This KD Tree only focus on LOCATION!!!
   #The mission of this KD TREE is to increase the coverage rate
   train_for_tree=[]
   for record in operator:
       sample_for_tree=[]
       sample_for_tree.append( record[7]/lat_scaler1 )
       sample_for_tree.append( record[8]/lon_scaler1 )
       train_for_tree.append( sample_for_tree )
   tree3=KDTree( train_for_tree )
   
   #Construction of Double Dictionary(First Key is the time sections ; Second Key is the destinations of the records which happen in each time section)
   double_dict=dict()
   double_dict[ 1 ] = dict()
   double_dict[ 1 ][ "sum" ]=0 
   
   double_dict[ 2 ] = dict()
   double_dict[ 2 ][ "sum" ]=0 
   
   double_dict[ 3 ] = dict()
   double_dict[ 3 ][ "sum" ]=0 
   
   double_dict[ 4 ] = dict()
   double_dict[ 4 ][ "sum" ]=0 
   
   double_dict[ 5 ] = dict()
   double_dict[ 5 ][ "sum" ]=0 
   
   double_dict[ 6 ] = dict()
   double_dict[ 6 ][ "sum" ]=0 
   
   double_dict[ 7 ] = dict()
   double_dict[ 7 ][ "sum" ]=0 
   
   for record in operator:
       if(record[2]>=6.0 and record[2]<9.5) :
     
           if record[6] not in double_dict[ 1 ]:
               double_dict[ 1 ][ record[6] ] = 1
               double_dict[ 1 ][ "sum" ] += 1
           else:
               double_dict[ 1 ][ record[6] ] += 1
               double_dict[ 1 ][ "sum" ] += 1
               
       elif (record[2]>=9.5 and record[2]<11.5):
      
           if record[6] not in double_dict[ 2 ]:
               double_dict[ 2 ][ record[6] ] = 1
               double_dict[ 2 ][ "sum" ] += 1
           else:
               double_dict[ 2 ][ record[6] ] += 1
               double_dict[ 2 ][ "sum" ] += 1
               
       elif (record[2]>=11.5 and record[2]<14):
    
           if record[6] not in double_dict[ 3 ]:
               double_dict[ 3 ][ record[6] ] = 1
               double_dict[ 3 ][ "sum" ] += 1
           else:
               double_dict[ 3 ][ record[6] ] += 1
               double_dict[ 3 ][ "sum" ] += 1
               
       elif (record[2]>=14.0 and record[2]<17.0):
            
           if record[6] not in double_dict[ 4 ]:
               double_dict[ 4 ][ record[6] ] = 1
               double_dict[ 4 ][ "sum" ] += 1
           else:
               double_dict[ 4 ][ record[6] ] += 1
               double_dict[ 4 ][ "sum" ] += 1
               
           
       elif (record[2]>=17.0 and record[2]<20.0):
         
           if record[6] not in double_dict[ 5 ]:
               double_dict[ 5 ][ record[6] ] = 1
               double_dict[ 5 ][ "sum" ] += 1
           else:
               double_dict[ 5 ][ record[6] ] += 1
               double_dict[ 5 ][ "sum" ] += 1
               
           
       elif (record[2]>=20.0 and record[2]<23.0):
        
           if record[6] not in double_dict[ 6 ]:
               double_dict[ 6 ][ record[6] ] = 1
               double_dict[ 6 ][ "sum" ] += 1
           else:
               double_dict[ 6 ][ record[6] ] += 1
               double_dict[ 6 ][ "sum" ] += 1
               
       else:
                     
           if record[6] not in double_dict[ 7 ]:
               double_dict[ 7 ][ record[6] ] = 1
               double_dict[ 7 ][ "sum" ] += 1
           else:
               double_dict[ 7 ][ record[6] ] += 1
               double_dict[ 7 ][ "sum" ] += 1
               
   
   #A combination of all the structures constructed above
   Knowledges=( namedict_or, namedict_ed, departdict_or, Popularity_Map_departure, Popularity_Map_destination, tree1, double_dict, tree2, tree3)
   
   del train_for_tree
   del namedict_or
   del namedict_ed
   del departdict_or
   del Popularity_Map_departure
   del Popularity_Map_destination
   del double_dict
   del tree1
   del tree2
   del tree3
   
   
   for real_record in operated:#operated can be interpreted as "test"
       """Construct Negative Examples For Each Record In The Operated Dataset"""
       negative_examples=[]
       Method1(real_record,negative_examples,Knowledges,operator) #The destination to which the user used to go(we do not fix the start)
       Method2(real_record,negative_examples,Knowledges,operator) #The start from which the user used to start(we do not fix the destination)
       Method3(real_record,negative_examples,Knowledges,operator) #The destination to which the user used to go in this given start (3 is in 1)
       Method4(real_record,negative_examples,Knowledges,operator) #The destination to which all the users used to go from this start
       Method5(real_record,negative_examples,Knowledges,operator) #The prediction made by Collaborative Fliter(Based on KDTree)
       Method6(real_record,negative_examples,Knowledges,operator,mode) #The prediction made by Round_Trip trick
       Method7(real_record,negative_examples,Knowledges,operator) #The prediction made by Naive Gaussian-Bayes Algorithm
       Method8(real_record,negative_examples,Knowledges,operator) #The prediction made by Collaborative Fliter(Based on KDTree)
       """Construct Some Features about the Collective Behaviours of all Negative Examples"""
       #Neighbour Counting
       l=len(negative_examples)
       Neighbour_Matrix=[ [0]*l for ww in range(l) ]
       for pp in range( l ): #Draw the Neighbour Matrix
           qq=pp+1
           while qq <= l-1:
               lat1,lng1=negative_examples[pp][-2],negative_examples[pp][-1]
               lat2,lng2=negative_examples[qq][-2],negative_examples[qq][-1]
               d=distan(lat1,lng1,lat2,lng2)
               if d < Neighbor_Condition:
                   Neighbour_Matrix[pp][qq]=Neighbour_Matrix[qq][pp]=1
               qq+=1
       for pp in range( l ):# Sum the neighbors signal over rows of Neighbour_Matrix
           del negative_examples[pp][-1]
           del negative_examples[pp][-1]
           negative_examples[pp].insert( len(negative_examples[pp])-1 , sum(Neighbour_Matrix[pp]) )
       
       #Group by destination of negative examples
       same_destination=dict()
       for ii in range(len(negative_examples)):
           example=negative_examples[ii]
           if example[-1] not in same_destination:
               same_destination[ example[-1] ]=[]
               same_destination[ example[-1] ].append( ii )
           else:#The list has already existed
               same_destination[ example[-1] ].append( ii )
       #Add One-Hot-Vector
       for des in same_destination:
           d=same_destination[ des ]
           if len( d ) == 1:
               pass
           else: #If the same destination appears several times , then we add their one hot vector
               m1=0
               m2=0
               m3=0
               m4=0
               m5=0
               m6=0
               m7=0
               m8=0
               for ii in d: # 5---11
                   m1+=negative_examples[ii][5]
                   m2+=negative_examples[ii][6]
                   m3+=negative_examples[ii][7]
                   m4+=negative_examples[ii][8]
                   m5+=negative_examples[ii][9]
                   m6+=negative_examples[ii][10]
                   m7+=negative_examples[ii][11]
                   m8+=negative_examples[ii][12]
               for ii in d:
                   negative_examples[ii][5]=m1
                   negative_examples[ii][6]=m2
                   negative_examples[ii][7]=m3
                   negative_examples[ii][8]=m4
                   negative_examples[ii][9]=m5
                   negative_examples[ii][10]=m6
                   negative_examples[ii][11]=m7
                   negative_examples[ii][12]=m8
               
                
       """Transform the Location-Prediction problem into Classification problem which XGboost can handle"""
       negative_examples=np.array( negative_examples , dtype=object )
       if mode=="train":
           lis=[ (1 if (des==real_record[3]) else 0) for des in negative_examples[ : , len(negative_examples[0])-1 ] ]
           X_train_xgboost.extend( negative_examples[ : , 0:len(negative_examples[0])-1 ] )
           Y_train_xgboost.extend( lis )

       else: #mode == "test"
           test_xgboost[ real_record[-1] ] = negative_examples
   
   if mode=="train":
       return np.array(X_train_xgboost) , Y_train_xgboost
   else:
       return test_xgboost
    
"""#####################################################################Main Function#####################################################################"""
"""#####################################################################Main Function#####################################################################"""
"""#####################################################################Main Function#####################################################################"""
"""#####################################################################Main Function#####################################################################"""


print("####################"+"Loading Data form disk")
"""Load Training Set(Stored in data_train)"""
# The Following is the content of elements in data_train
# 0. User Name
# 1. Wether_or_not the date of the record is weekend
# 2. The specific time of the record
# 3. Departure point(Geohash)
# 4. Latitude of Departure point
# 5. Longitude of Departure point
# 6. Destination(Geohash)
# 7. Latitude of Destination
# 8. Longitude of Destination
# 9. Orderid
file_name_train="train.csv"
dataframe_train=read_csv(file_name_train,usecols=[0,1,4,5,6])
dataframe_train=dataframe_train.sort_index(by='starttime') # Sort the training Dataset by date
data_train_raw=dataframe_train.values
data_train=[]
for train in data_train_raw:#Rearrange the order of features
    lat1,lng1,a,b=decode_exactly( train[3] )
    lat2,lng2,a,b=decode_exactly( train[4] )
    train_sample=(train[1], work_play(train[2]), hour_minute(train[2]), train[3], float(lat1), float(lng1), train[4], float(lat2), float(lng2),train[0])
    data_train.append(train_sample)
del file_name_train
del dataframe_train
del data_train_raw
data_train=np.array(data_train,dtype=object) #data_train is a numpy array which can be sliced, masked
                                             #but can not use the append and extend function.
                                             #The reason why we assert dtype=object is that we want a mixed type array



steps=np.array([262569,272210,265173,225281,236594,279554,288719,322201,314516,134159,209440,124816,150456,128408])
step=steps.cumsum()
#step=len(data_train)//Q
#Construct operated dataset , of which we generate the negative examples
#We can comprehend the operated set as the "test" set
#a subset of data_train
#Construct operator dataset , from which the information is extracted to build operated dataset
#We can comprehend the operator set as the "train" set
#a subset of data_train
operated=data_train[step[7]:]
operator=data_train[:step[7]]
operator=np.array(operator, dtype=object)
operated=np.array(operated, dtype=object)
"""Construct Negative Examples"""
X,Y = Negative_Examples(operated,operator,"train")
del operated
del operator
#Write the X into disk
name_X="CV_X"+".npy"
np.save(name_X, X)
        
#Write the Y into disk
name_Y="CV_Y"+".npy"
np.save(name_Y,Y)

del X
del Y
   
   
"""Load the Negative Examples into Python"""
"""The Reason why we save the negative examples into the disk is to save memory"""
X_train_xgboost=[] #This is the input train set with negative examples
Y_train_xgboost=[] #This is the output train set with negative examples
#Construct X_train_xgboost
#X_train_xgboost is a numpy array after construction
ar = np.load('CV_X.npy')
X_train_xgboost.extend(ar)
del ar

X_train_xgboost=np.array(X_train_xgboost)

#Construct Y_train_xgboost
dr = np.load('CV_Y.npy')
dr=list(dr)
Y_train_xgboost.extend(dr)
del dr


"""Data Preparation for XGBoost training"""
#Split training_dataset into old and new clients
X_old_train = []
Y_old_train = []
X_new_train = []
Y_new_train = []
for i in range( len( Y_train_xgboost ) ):
    if X_train_xgboost[i][0] == 1: #Old
        X_old_train.append(X_train_xgboost[i])
        Y_old_train.append(Y_train_xgboost[i])
    else:
        X_new_train.append(X_train_xgboost[i])
        Y_new_train.append(Y_train_xgboost[i])
del X_train_xgboost
del Y_train_xgboost


"""Train XGBoost and Hyperparameters Tuning(Old Clients)"""
"""Note: the training process is completed on the Old Clients' dataset"""
print("####################"+"Train XGboost and Hyperparameter Tuning(Old Clients)")
seed(1)
x,y=random_subset(X_old_train,Y_old_train) #x and y are used for Hyperparameter Tuning
x,all_x,all_y=np.array(x),np.array(X_old_train),Y_old_train#Change the structure into Numpy Array 
del X_old_train
del Y_old_train
dtrain = xgb.DMatrix(all_x,all_y)
del all_x
del all_y
##input train_x as x ,train_y as y,test_x as test_X
param_test1 = {
 'max_depth':[i for i in range(1,10,1)],  ##use loop to extract the value
 'min_child_weight':[i for i in range(10,60,5)]  ##如果出现两个参数，会取其组合的最优解
}
print("####################"+"Grid-Search 1")
gsearch1 = grid_search.GridSearchCV(
estimator = XGBClassifier(
learning_rate =0.1,
n_estimators=140, max_depth=3,
min_child_weight=2,
gamma=0,
subsample=0.8,
colsample_bytree=1,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27),
param_grid = param_test1,
scoring='roc_auc',
n_jobs=4,
iid=False,
cv=5)
gsearch1.fit(x,y)
gsearch1.grid_scores_, gsearch1.best_params_,gsearch1.best_score_
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
print("####################"+"Grid-Search 2")
gsearch3 = grid_search.GridSearchCV(
estimator = XGBClassifier( 
learning_rate =0.1, 
n_estimators=140, 
max_depth=gsearch1.best_params_['max_depth'], 
min_child_weight=gsearch1.best_params_['min_child_weight'], 
gamma=0.3, 
subsample=0.8, 
colsample_bytree=0.8, 
objective= 'binary:logistic', 
nthread=4, 
scale_pos_weight=1,
seed=27), 
param_grid = param_test3, 
scoring='roc_auc',
n_jobs=4,
iid=False, 
cv=5)
gsearch3.fit(x,y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
print("####################"+"Grid-Search 3")
gsearch4 = grid_search.GridSearchCV(
estimator = XGBClassifier(
learning_rate =0.1,
n_estimators=177,
max_depth=gsearch1.best_params_['max_depth'],
min_child_weight=gsearch1.best_params_['min_child_weight'],
gamma=gsearch3.best_params_['gamma'],
subsample=0.7,
colsample_bytree=0.6,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27),
param_grid = param_test4,
scoring='roc_auc',
n_jobs=4,
iid=False,
cv=5)
gsearch4.fit(x,y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
param_test6 = {
 'reg_alpha':[i/10 for i in range(5,15)]
}
#param_test6 = {
 ##'reg_alpha':[i/1000 for i in range(10)]
print("####################"+"Grid-Search 4")
gsearch6 = grid_search.GridSearchCV(
estimator = XGBClassifier(
learning_rate =0.1,
n_estimators=177,
max_depth=gsearch1.best_params_['max_depth'],
min_child_weight=gsearch1.best_params_['min_child_weight'],
gamma=gsearch3.best_params_['gamma'],
subsample=gsearch4.best_params_['subsample'],
colsample_bytree=gsearch4.best_params_['colsample_bytree'],
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27),
param_grid = param_test6,
scoring='roc_auc',
n_jobs=4,
iid=False,
cv=5)
gsearch6.fit(x,y)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
params={
'booster':'gbtree',
'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
#'nthread':7,# cpu 线程数 默认最大
'eta': 0.01, # 如同学习率
'min_child_weight':gsearch1.best_params_['min_child_weight'], 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'max_depth':gsearch1.best_params_['max_depth'], # 构建树的深度，越大越容易过拟合
'gamma':gsearch3.best_params_['gamma'],  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
'subsample':gsearch4.best_params_['subsample'], # 随机采样训练样本
'colsample_bytree':gsearch4.best_params_['colsample_bytree'], # 生成树时进行的列采样 
'lambda':gsearch6.best_params_['reg_alpha'],  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#'alpha':0, # L1 正则项参数
#'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
'objective': 'binary:logistic', #二分类的问题
#'num_class':10, # 类别数，多分类与 multisoftmax 并用
'seed':1000, #随机种子a
'eval_metric': 'auc'
}
plst = list(params.items())
print("####################"+"Training")
model_old = xgb.train(plst,dtrain,num_boost_round=100)
del x
del y


"""Train XGBoost and Hyperparameters Tuning(New Clients)"""
"""Note: the training process is completed on the New Clients' dataset"""
print("####################"+"Train XGboost and Hyperparameter Tuning(New Clients)")
x,y=random_subset(X_new_train,Y_new_train) #x and y are used for Hyperparameter Tuning
x,all_x,all_y=np.array(x),np.array(X_new_train),Y_new_train#Change the structure into Numpy Array
del X_new_train
del Y_new_train
dtrain = xgb.DMatrix(all_x,all_y)
del all_x
del all_y
##input train_x as x ,train_y as y,test_x as test_X
param_test1 = {
 'max_depth':[i for i in range(1,10,1)],  ##use loop to extract the value
 'min_child_weight':[i for i in range(10,60,5)]  ##如果出现两个参数，会取其组合的最优解
}
print("####################"+"Grid-Search 1")
gsearch1 = grid_search.GridSearchCV(
estimator = XGBClassifier(
learning_rate =0.1,
n_estimators=140, max_depth=3,
min_child_weight=2,
gamma=0,
subsample=0.8,
colsample_bytree=1,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27),
param_grid = param_test1,
scoring='roc_auc',
n_jobs=4,
iid=False,
cv=5)
gsearch1.fit(x,y)
gsearch1.grid_scores_, gsearch1.best_params_,gsearch1.best_score_
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
print("####################"+"Grid-Search 2")
gsearch3 = grid_search.GridSearchCV(
estimator = XGBClassifier( 
learning_rate =0.1, 
n_estimators=140, 
max_depth=gsearch1.best_params_['max_depth'], 
min_child_weight=gsearch1.best_params_['min_child_weight'], 
gamma=0.3, 
subsample=0.8, 
colsample_bytree=0.8, 
objective= 'binary:logistic', 
nthread=4, 
scale_pos_weight=1,
seed=27), 
param_grid = param_test3, 
scoring='roc_auc',
n_jobs=4,
iid=False, 
cv=5)
gsearch3.fit(x,y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
print("####################"+"Grid-Search 3")
gsearch4 = grid_search.GridSearchCV(
estimator = XGBClassifier(
learning_rate =0.1,
n_estimators=177,
max_depth=gsearch1.best_params_['max_depth'],
min_child_weight=gsearch1.best_params_['min_child_weight'],
gamma=gsearch3.best_params_['gamma'],
subsample=0.7,
colsample_bytree=0.6,
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27),
param_grid = param_test4,
scoring='roc_auc',
n_jobs=4,
iid=False,
cv=5)
gsearch4.fit(x,y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
param_test6 = {
 'reg_alpha':[i/10 for i in range(5,15)]
}
#param_test6 = {
 ##'reg_alpha':[i/1000 for i in range(10)]
print("####################"+"Grid-Search 4")
gsearch6 = grid_search.GridSearchCV(
estimator = XGBClassifier(
learning_rate =0.1,
n_estimators=177,
max_depth=gsearch1.best_params_['max_depth'],
min_child_weight=gsearch1.best_params_['min_child_weight'],
gamma=gsearch3.best_params_['gamma'],
subsample=gsearch4.best_params_['subsample'],
colsample_bytree=gsearch4.best_params_['colsample_bytree'],
objective= 'binary:logistic',
nthread=4,
scale_pos_weight=1,
seed=27),
param_grid = param_test6,
scoring='roc_auc',
n_jobs=4,
iid=False,
cv=5)
gsearch6.fit(x,y)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
params={
'booster':'gbtree',
'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
#'nthread':7,# cpu 线程数 默认最大
'eta': 0.01, # 如同学习率
'min_child_weight':gsearch1.best_params_['min_child_weight'], 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'max_depth':gsearch1.best_params_['max_depth'], # 构建树的深度，越大越容易过拟合
'gamma':gsearch3.best_params_['gamma'],  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
'subsample':gsearch4.best_params_['subsample'], # 随机采样训练样本
'colsample_bytree':gsearch4.best_params_['colsample_bytree'], # 生成树时进行的列采样 
'lambda':gsearch6.best_params_['reg_alpha'],  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#'alpha':0, # L1 正则项参数
#'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
'objective': 'binary:logistic', #二分类的问题
#'num_class':10, # 类别数，多分类与 multisoftmax 并用
'seed':1000, #随机种子a
'eval_metric': 'auc'
}
plst = list(params.items())
print("####################"+"Training")
model_new = xgb.train(plst,dtrain,num_boost_round=100)
del x
del y


"""Load Testing Set(Stored in data_test)"""
# The Following is the content of elements in data_test
# 0. User Name
# 1. Wether_or_not the date of the record is weekend
# 2. The specific time of the record
# 3. Departure point(Geohash)
# 4. Latitude of Departure point
# 5. Longitude of Departure point
# 6. Order ID(For submission request)
# Note: The elements of data_train and data_test share the same formation in the first 6 entries. 
file_name_test="test.csv"
dataframe_test=read_csv(file_name_test,usecols=[0,1,4,5])
data_test_raw=dataframe_test.values  
del dataframe_test
data_test=[] 
for test in data_test_raw:
    lat1,lng1,a,b=decode_exactly( test[3] )
    test_sample=(test[1], work_play(test[2]), hour_minute(test[2]), test[3], float(lat1), float(lng1), test[0])
    data_test.append(test_sample)
del file_name_test
del data_test_raw  
data_test=np.array(data_test,dtype=object) 


"""Construct Negative Examples of Test set"""
print("####################"+"Construct Negative Examples of Test set")
d0=Negative_Examples(data_test,data_train,"test")
np.save('test_dictionary.npy',d0)
del data_train
del data_test


"""Make Prediction By Trained XGboost"""
print("####################"+"Make Prediction")
submission=[]
for orderid in d0:
    negatives=d0[ orderid ]
    test_x=negatives[ : , 0:len(negatives[0])-1 ]
    test_y=np.array([])
    if test_x[0][0]==1:#old clients
        test_y = model_old.predict(xgb.DMatrix(test_x),ntree_limit=0)  ##test output
    else:#test_x[0][0]==0 new clients
        test_y = model_new.predict(xgb.DMatrix(test_x),ntree_limit=0)  ##test output
    
    prediction_destination=dict()
    for ii in range(len(negatives)):
        if negatives[ii][-1] not in prediction_destination:
            prediction_destination[ negatives[ii][-1] ]=test_y[ii]
        else:
            prediction_destination[ negatives[ii][-1] ]+=test_y[ii]
    prediction_destination=sorted(prediction_destination.items(),key=lambda d:d[1],reverse=True)
    
    result=[]
    result.append( orderid )
    for i in range(N):
        result.append( prediction_destination[i][0] )
        
    submission.append(result)
del d0


"""Print the prediction-result to file"""
#Sort the submission to fit the submission request
print("####################"+"Submission")
with open("submission.csv","w",newline='') as file:
    writer=csv.writer(file)
    for row in submission:
        writer.writerow(row)   