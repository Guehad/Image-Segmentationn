from PIL import Image
import scipy.io
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import toimage
from scipy.misc import imresize
import cv2
from sklearn.metrics import f1_score
from sklearn.cluster import SpectralClustering
import random
from scipy.io import loadmat

def load_image(image_name):
    print('C:/Users/DELL/Documents/Pattern Recognition/ass2/BSR/BSDS500/data/images/train/'+image_name)
    img = Image.open('C:/Users/DELL/Documents/Pattern Recognition/ass2/BSR/BSDS500/data/images/train/'+image_name)
    mat = loadmat('C:/Users/DELL/Documents/Pattern Recognition/ass2/BSR/BSDS500/data/groundTruth/train/'+image_name[:-4]+'.mat')
    ground_truth = mat['groundTruth']
    return ground_truth, img


def show_segment(image_name):
    ground_truth, img = load_image(image_name)
    img.show()
       
    for i in range(ground_truth.shape[1]):
        segment = ground_truth[0, i]
        print(segment)
        print('segment printed')
        bound = segment[0, 0][1]
        print(bound.shape)
        print('bound printed sawsaw')
        bound=np.where(bound==0,255,bound)
        ff=toimage(bound)
        plt.figure(i + 1)
        plt.imshow(ff)
    plt.show()
    
def conditional_entropy(seg, gt):
    
    clusters = np.unique(seg).tolist()
    gt_clusters = np.unique(gt)
    H = 0
    # for every cluster in seg
    for cluster in clusters:
        Hi = 0
        indices = np.where(seg == cluster)
        partitions = gt[indices]
        # for every cluster in gt
        for gt_cluster in gt_clusters:
            nij = partitions[partitions == gt_cluster].shape[0]
            ni =  indices[0].shape[0]
            if nij / ni != 0:
                Hi += (nij / ni) * np.log2(nij / ni)
        Hi *= -1

        H += len(indices)*Hi/len(seg)
    return H

directory='C:/Users/DELL/Documents/Pattern Recognition/ass2/BSR/BSDS500/data/images/train'
labelx=[]
image=[]
rgb=[]
imnames=[]
for f in os.listdir(directory):
    imnames.append(f)
   # print(f)
   # print(os.path.join(directory, f))
    #labelx.append(show_segment(f))


#print("Conditional Entropy :",conditional_entropy(np.asarray(label),true) )  
#print("Conditional Entropy :",conditional_entropy(label,bound) ) 
#print('************************')
    
#def f1_score_scratch(seg, gt):
    #clusters = np.unique(seg)
    #F = 0
    #for i in range(clusters.shape[0]):
        #indices = np.where(seg == clusters[i])
        #partitions, pcounts = np.unique(gt[indices], return_counts=True)
        #precision = np.max(pcounts) / partitions.shape[0]
        ## Recall is the ratio of correctly predicted positive observations to 
        ## the all observations in actual class
        #max_value_index = np.argmax(pcounts) # the index of the most occurred value.
        #max_value = partitions[max_value_index] # the most occurred value.
        #max_value_occurrences = gt[gt==max_value].shape[0]
        #recall = (np.max(pcounts)) / max_value_occurrences

        #F += (2 * precision * recall) / (precision + recall)

    #F /= clusters.shape[0]
    #return F


def kmeans(k,image):
    
    
    image=np.asarray(image)
    vectorized=image.reshape(-1,3)
    vectorized=np.float32(vectorized)
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10, 1.0)
    ret,label,center=cv2.kmeans(vectorized,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    res = center[label.flatten()]
    segmented_image = res.reshape((image.shape))
    return label.reshape((image.shape[0],image.shape[1])),segmented_image.astype(np.uint8)


###k-meaaaaaaans fmeasure 
#ground_truth, img = load_image()
#img.show()
##img=np.asarray(img)
##cv2.imshow("input",img)
#K=[3,5,7,9,11]
#for ks in K:

    #label,result=kmeans(ks,img)
    #img=np.asarray(img)
    #cv2.imshow("segmented",result)
    #cv2.imwrite('mmg1.png',result)
    ##print(label)
    #totalF=0
    #cv2.waitKey(0)
    #print(np.asarray(ground_truth).shape)
    #SegmentN=np.asarray(ground_truth).shape[1]
    #print((label).shape)
    #for i in range(ground_truth.shape[1]):
        
       ## print((ground_truth[:,:,0]).shape)
        ##b=ground_truth[0,0]
        ##print(b.shape)
        #s=ground_truth[0,i]
        #bound = s[0, 0][0]
        #print(bound)
        #print(bound.shape)
        #print('el fscore')
         
        #print('mas3oud measure')
        ##print(f1_score_single(bound.ravel(),label.ravel()))
        ##print(f_measure(label.ravel(),bound.ravel(),ks))
        ##unique dah 3shan fi 7agat msh ma3molaha predict aslan fa betdrab fl average calculation.
        #Fscr=f1_score(bound.ravel(), label.ravel(), labels=np.unique(label.ravel()), average='weighted', sample_weight=None)
        #totalF=totalF+Fscr
    #AvgF=totalF/SegmentN 
    #print('K is ', ks)
    #print('Average Score for this K is' , AvgF)
    

def ncut(k,img) :
    img = imresize(img, 0.3) / 255
    n = img.shape[0]
    m = img.shape[1]
    print(n,m)
    img = (np.asarray(img)).reshape(-1,3)
    print(img.shape)
    clustering = SpectralClustering(n_clusters=k,
                                  affinity='nearest_neighbors',
                                  gamma=1,
                                  n_neighbors=k,
                                  n_jobs=-1,
                                      eigen_solver='arpack'                                  
                                  )
    
    labelss=clustering.fit_predict(np.float32(img))
   
    labelss = (labelss.reshape(n, m))
    print("here",labelss)
    print(clustering.labels_)
    plt.figure(figsize=(12, 12))
    #plt.axis('off')
    plt.imshow(labelss)
    plt.show()
    return  clustering.labels_,n,m

#N-cut

ground_truth, img = load_image()
img.show()
label,result1,result2=ncut(5,img)
img=np.asarray(img)
#cv2.imshow("segmented",result)
#print(label)
#cv2.waitKey(0)
print(np.asarray(ground_truth).shape)
SegmentN=np.asarray(ground_truth).shape[1]
print(SegmentN)
print((label).shape)
totalF=0
totalEn=0
for i in range(ground_truth.shape[1]):
    
   # print((ground_truth[:,:,0]).shape)
    #b=ground_truth[0,0]
    #print(b.shape)
    s=ground_truth[0,i]
    bound = s[0, 0][0]
    print(bound)
    print(bound.shape)
    print('el fscore')
    print('mas3oud measure')
    true=np.resize(bound,result1*result2) 
    #print(f1_score_single(bound.ravel(),label.ravel()))
    #print(f_measure(label.ravel(),bound.ravel(),ks))
    #unique dah 3shan fi 7agat msh ma3molaha predict aslan fa betdrab fl average calculation.
    print("Conditional Entropy :",conditional_entropy(np.asarray(label),true))
    Fscr=f1_score(true, label, labels=np.unique(label), average='weighted', sample_weight=None)
    ConEnt=conditional_entropy(np.asarray(label),true)
    print(Fscr)
    print(ConEnt)
    totalF=totalF+Fscr
    totalEn=totalEn+ConEnt
    
print('eltotal kolo', totalF)
AvgF=totalF/SegmentN
AvgEnt=totalEn/SegmentN
#print('K is ', ks)
print('Average Score for this K is' , AvgF)
print('Average Entropy for this K is',AvgEnt)

def big_picture ():
    i=0
    while i<5:
        i+=1
        image=random.choice(imnames)
        show_segment(image)
        ground_truth,image=load_image(image)
        ncut(5,image)
        l,result =kmeans(5,image)
        cv2.imshow("segmented",result)
    return
#big_picture()
show_segment()