package com.ysc.service.impl;

import com.ysc.mapper.SiteMapper;
import com.ysc.pojo.CloudCabinet;
import com.ysc.pojo.Shop;
import com.ysc.service.SiteService;
import org.gavaghan.geodesy.Ellipsoid;
import org.gavaghan.geodesy.GeodeticCalculator;
import org.gavaghan.geodesy.GeodeticCurve;
import org.gavaghan.geodesy.GlobalCoordinates;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class SiteServiceImpl implements SiteService {
    @Autowired
    private SiteMapper siteMapper;
    @Override
    public List<CloudCabinet> terminalShow() {
        return siteMapper.selectTerminalAll();
    }

    @Override
    public List<Integer> getTerminalDistanceMeterList(Double longitude, Double latitude) {
        System.out.println(longitude);

        GlobalCoordinates source = new GlobalCoordinates(longitude, latitude);
        List<CloudCabinet> cloudCabinetList= siteMapper.selectTerminalAll();
        List<Integer> distanceList=new ArrayList<>();
        for(CloudCabinet cloudCabinet : cloudCabinetList){
            GlobalCoordinates target = new GlobalCoordinates(cloudCabinet.getLongitude(), cloudCabinet.getLatitude());
            Integer meter = getDistanceMeter(source, target, Ellipsoid.WGS84);
//            System.out.println(meter);
            distanceList.add(meter);
        }
        return distanceList;
    }

    @Override
    public List<Shop> shopShow() {
        return siteMapper.selectShopAll();
    }

    @Override
    public CloudCabinet getTerminalAddress(Integer id) {
        return siteMapper.selectTerminalById(id);
    }

    @Override
    public List<Integer> getShopDistanceMeterList(Double longitude, Double latitude) {
        GlobalCoordinates source = new GlobalCoordinates(longitude, latitude);
        List<Shop> shopList= siteMapper.selectShopAll();
        List<Integer> distanceList=new ArrayList<>();
        for(Shop shop : shopList){
            GlobalCoordinates target = new GlobalCoordinates(shop.getLongitude(), shop.getLatitude());
            Integer meter = getDistanceMeter(source, target, Ellipsoid.WGS84);
//            System.out.println(meter);
            distanceList.add(meter);
        }
        return distanceList;
    }

    public static Integer getDistanceMeter(GlobalCoordinates gpsFrom, GlobalCoordinates gpsTo, Ellipsoid ellipsoid)
    {
        //创建GeodeticCalculator，调用计算方法，传入坐标系、经纬度用于计算距离
        GeodeticCurve geoCurve = new GeodeticCalculator().calculateGeodeticCurve(ellipsoid, gpsFrom, gpsTo);

        return (int) geoCurve.getEllipsoidalDistance();
    }
}
