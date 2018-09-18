# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 16:42:38 2018

@author: zhangyaxu
"""

import numpy as npy
import math
import pandas as pds
import random

def get_lnglat_data():
    '''
    此处给出 传入算法的数据的结构。使用时按实际需求清洗数据即可。
    '''
    df=pds.DataFrame({'loc_id':[1,2],'lng':[112.323,114.632],'':[33.2243,43.143]})
    return df
    
def pd_lnglat_meanshift_cluster(pd_df,bandwidth,eps,max_iteration=10,center_default_mode='random',min_num=2):
    '''
    基于meanshift原理,代入自定义的haversine距离算子， 对地理经纬度聚类
    pd_df : pandas-dataframe  data  which columns should be  [loc_id,lng,lat] 即 [编号,经度,纬度] , 
            编号必须是unique的，可以是订单id ,用户id ，甚至无实际场景意义的一串序号id，经纬度必须是[-180,180] [-90,90]区间形式
    bandwidth ：控制类的半径 bandwidth
    eps ： 更新质心时，质心的变化距离的最大容忍度阈值，当<= eps，停止更新质心，输出新类
    max_iteration ：迭代更新质心时，容忍的最大迭代次数
    center_default_mode : 每次新的类 的质心的 生成方式 ,[random,hot_random] ==> [全域随机,密集域随机]
    min_num : 初始化质心时，允许质心周边的最少点个数，若<min_num,则认为点过少，不具备群体性，不予聚类；
              此参数仅在hot_random 模式下存在作用，且一旦<min_num,hot_random会停止之后的所有聚类判断，所以 hot_random模式也会效率更高 
              （得益于 hot_random是从密集区域开始初始化中心，之后的区域会越来越不密集）
    '''
    # 自定义 haversine算法
    # 单独的一组数值
    def lng_lat_distince(lng1,lat1,lng2,lat2):
        '''
        Use Haversine distince formula
        输入 ： 球体表面两点的 经纬度 [lng1,lat1]  [lng2,lat2]  (算法中 会把经纬度转化为弧度)
        输出 d ： 半径为1的球体上 ，两个弧度标识的点的距离 
        '''
        # 务必 把[-180,180]  [-90,90]的经纬度的数值度degrees格式，转化为 三角函数的弧度radians
        lng1,lat1,lng2,lat2=lng1*math.pi/180.0,lat1*math.pi/180.0,lng2*math.pi/180.0,lat2*math.pi/180.0
        dlat=lat1-lat2
        dlng=lng1-lng2
        H_dr=math.sin(dlat/2.0)**2+math.cos(lat1)*math.cos(lat2)*(math.sin(dlng/2.0)**2)
        R=1.0  # R是球体半径 ,默认是1.0
        d=math.asin(math.sqrt(H_dr))*2.0*R   #  这个 是根据 又有在球体上： H_dr=Haversine(d/R) 反推出来的d  ，
        return d
    
    # pandas版,尽量使用矢量运算，提高效率
    def lng_lat_distince2(lnglat_df,lng2,lat2):
        '''
        Use Haversine distince formula
        输入 ： 球体表面两点的 经纬度 lnglat_df=df([lng1,lat1])  [lng2,lat2]  (算法中 会把经纬度转化为弧度)
        输出 d ： 半径为1的球体上 ，两个弧度标识的点的距离 
        '''
        # 务必 把[-180,180]  [-90,90]的经纬度的数值度degrees格式，转化为 三角函数的弧度radians
        lnglat_df,lng2,lat2=lnglat_df*math.pi/180.0,lng2*math.pi/180.0,lat2*math.pi/180.0
        dlat=lnglat_df.lat-lat2
        dlng=lnglat_df.lng-lng2
        H_dr=npy.sin(dlat/2.0)**2+npy.cos(lnglat_df.lat)*math.cos(lat2)*(npy.sin(dlng/2.0)**2)
        R=1.0  # R是球体半径 ,默认是1.0
        d=npy.arcsin(npy.sqrt(H_dr))*2.0*R   #  这个 是根据 又有在球体上： H_dr=Haversine(d/R) 反推出来的d  ，
        return d
    
    # 初始化 中心 函数
    def generate_center(dfx,center_default_mode='random',min_num=min_num):
        if center_default_mode=='random':
            try:
                center_lnglat=dfx.loc[random.randint(0,dfx.shape[0]-1),['lng','lat']].values.tolist()  
            except:
                center_lnglat=[]
            current_num=min_num
        elif center_default_mode=='hot_random':
            cols_exist=dfx.columns
            if 'lng_group' not in cols_exist:
                dfx['lng_group']=npy.round(dfx.lng,2)
                dfx['lat_group']=npy.round(dfx.lat,2)
            try:
                temg=dfx.groupby(['lng_group','lat_group'])[dfx.columns[0]].count().sort_values(ascending=False).reset_index()
                center_lnglat=dfx.loc[(dfx.lng_group==temg.iloc[0,0])&(dfx.lat_group==temg.iloc[0,1]),['lng','lat']].iloc[0,:].values.tolist()
                current_num=temg.iloc[0,2]
            except:
                center_lnglat=[]
                current_num=min_num
        return center_lnglat,dfx,current_num
    
    initial_center,dfnew,current_num=generate_center(dfx=pd_df.copy(),center_default_mode=center_default_mode,min_num=min_num) 
    cols=dfnew.columns  
    iter_k=0   # 每次 生成了新的聚类组 都要再次把迭代初始为0，供下一个类的迭代使用
    center0=initial_center
    current_label=1  # 聚类 序号 从 1 开始 ，每生成一个新类，新类获得current_label ，之后，current_label+1 进入下一次选组
    while center0!=[] and current_num>=min_num :
        # 迭代运算，计算 距离， 效率慢
        #clusterx=dfnew.apply(lambda x: pds.Series(x.values.tolist()+[lng_lat_distince(lng1=x[1],lat1=x[2],lng2=center0[0],lat2=center0[1])],cols.tolist()+['ds']),axis=1)
        #clusterx=clusterx.where(clusterx.ds<=bandwidth).dropna().reset_index(drop=True)
        # 矢量运算  计算距离
        clusterx=dfnew.copy()
        clusterx['ds']=lng_lat_distince2(lnglat_df=dfnew[['lng','lat']].copy(),lng2=center0[0],lat2=center0[1])
        clusterx=clusterx.loc[clusterx.ds<=bandwidth,:].copy().reset_index(drop=True)
        center1=clusterx[['lng','lat']].mean(axis=0).values.tolist()# 计算新的质心 center1
        #print(center0,center1)
        center_eps=lng_lat_distince(lng1=center1[0],lat1=center1[1],lng2=center0[0],lat2=center0[1])  # 计算 新旧质心的距离
        # 判断是否满足质心收敛到容忍阈值内 ，
        if center_eps<=eps or iter_k==max_iteration:
            if current_label%100.0==1:
                print('###### \n ',current_label,iter_k,center0,dfnew.shape,clusterx.shape,current_num)
            # 若满足收敛或达到最大迭代次数，则输出组，并从源数据组中去除当前组中的loc_id对应的数据记录
            if current_label==1:
                clustery=clusterx.iloc[:,0:3].copy()
                clustery['lng_center'],clustery['lat_center'],clustery['label_id']=center0[0],center0[1],current_label  # 第一个组 直接保留. [loc_id,lng,lat,lng_center,lat_center,label_y]
            else:
                clusterx2=clusterx.iloc[:,0:3].copy()
                clusterx2['lng_center'],clusterx2['lat_center'],clusterx2['label_id']=center0[0],center0[1],current_label
                clustery=pds.concat([clustery.copy(),clusterx2.copy()],axis=0).reset_index(drop=True)  # 若不是第一组 则 与 之前的  concat
            dfnew=dfnew.loc[dfnew.loc_id.isin(clusterx.loc_id)==False,:].reset_index(drop=True).copy() # 去除新聚类的组中的loc_id对应的数据记录
            center0,dfnew,current_num=generate_center(dfx=dfnew.copy(),center_default_mode=center_default_mode,min_num=min_num) # 对 更新的数据 再初始化质心
            iter_k=0 # 新类 计算 ，迭代重新从0开始
            current_label=current_label+1  # 聚类序号 增1
        else:
            iter_k=iter_k+1 #继续迭代 
            center0=center1 # 更新质心，使center0保留最新的质心 
    if current_num<min_num:
        dfnew['label_id']=dfnew.index+current_label
        dfnew['lng_center']=dfnew.lng.copy()
        dfnew['lat_center']=dfnew.lat.copy()
        clustery=pds.concat([clustery.copy(),dfnew.loc[:,cols[0:3].tolist()+[x+'_center' for x in cols[1:3]]+['label_id']].copy()],axis=0).reset_index(drop=True)
        current_label=dfnew.shape[0]+current_label-1
    
    return clustery,current_label

if __name__=='__main__':
    R=6371.0
    pd_df=get_lnglat_data()
    cluster_result,current_label=pd_lnglat_meanshift_cluster(pd_df=pd_df.copy(),bandwidth=1.0/R,eps=0.2/R,max_iteration=5,center_default_mode='hot_random',min_num=2)
    cluster_result.columns
    
    >> [loc_id,lng,lat,lng_center,lat_center,label_id]
    
    
