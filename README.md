# -Clustering-Algorithm
Gathering some  unusual  clustering  demand , also include some optimization of  popular algorithms 
主要是处理一些特殊的聚类需求，或者 改进一些流行的聚类算法  

**1. 经纬度聚类  longitude-latitude-clustering**  
对经纬度聚类，目标是 固定半径内的点聚类，采用meanshift的逻辑+使用haversine距离算子。   
若不固定聚类结果的半径，是基于密度可达的密集区域，则直接使用DBSCAN，metric参数设置为haversine即可 。    
> **注意：**  
> DBSCAN是不返回类中心的  ，且sklearn自带的 haversine距离算子要求传入的 [x,y]的顺序是[lat,lng]    
> meanshift原本是应用在图像识别领域，有固定的核函数，遂这里应用到经纬度时，需要自己构建代码，在迭代过程中加入haversine距离算子。  
