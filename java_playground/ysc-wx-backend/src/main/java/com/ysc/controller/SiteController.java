package com.ysc.controller;


import com.ysc.pojo.CloudCabinet;
import com.ysc.pojo.Result;
import com.ysc.pojo.Shop;
import com.ysc.service.SiteService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/site")
@Slf4j
public class SiteController {
    @Autowired
    private SiteService siteService;

    @GetMapping("/terminal")
    public Result terminalList(){
        List<CloudCabinet> terminalList= siteService.terminalShow();
        return Result.success(terminalList);
    }

    @GetMapping("/terminal/distance")
    public Result terminalDistanceList(Double longitude, Double latitude){
        List<Integer> distanceList= siteService.getTerminalDistanceMeterList(longitude,latitude);
        return Result.success(distanceList);
    }

    @GetMapping("/shop/distance")
    public Result shopDistanceList(Double longitude, Double latitude){
        List<Integer> distanceList= siteService.getShopDistanceMeterList(longitude,latitude);
        return Result.success(distanceList);
    }

    @GetMapping("/shop")
    public Result shopList(){
        List<Shop> shopList= siteService.shopShow();
        return Result.success(shopList);
    }

    @GetMapping("/terminal/{id}")
    public Result terminalShow(@PathVariable Integer id){
        CloudCabinet cloudCabinet= siteService.getTerminalAddress(id);
        return Result.success(cloudCabinet);
    }

}
