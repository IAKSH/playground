package com.ysc.service;

import com.ysc.pojo.CloudCabinet;
import com.ysc.pojo.Shop;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public interface SiteService {
    List<CloudCabinet> terminalShow();

    List<Integer> getTerminalDistanceMeterList(Double longitude, Double latitude);

    List<Shop> shopShow();

    CloudCabinet getTerminalAddress(Integer id);

    List<Integer> getShopDistanceMeterList(Double longitude, Double latitude);
}
