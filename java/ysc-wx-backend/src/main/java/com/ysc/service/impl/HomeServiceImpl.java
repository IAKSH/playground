package com.ysc.service.impl;

import com.ysc.mapper.HomeMapper;
import com.ysc.pojo.HanFu;
import com.ysc.pojo.HanFuImage;
import com.ysc.pojo.Shop;
import com.ysc.service.HomeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class HomeServiceImpl implements HomeService {

    @Autowired
    private HomeMapper homeMapper;
    @Override
    public List<HanFu> hanFuAllShow() {
        return homeMapper.selectAll();
    }

    @Override
    public Shop shopNameShow(Integer shopId) {
        return homeMapper.selectShopById(shopId);
    }

    @Override
    public List<HanFuImage> hanFuImageShow(Integer hanFuId) {
        return homeMapper.selectImageById(hanFuId);
    }

    @Override
    public HanFu hanFuShow(Integer hanFuId) {
        HanFu hanFu= homeMapper.selectHanFuInfoById(hanFuId);
        int views=hanFu.getViews()+1;
        homeMapper.upViews(views,hanFuId);
        return homeMapper.selectHanFuInfoById(hanFuId);
    }

    @Override
    public List<HanFu> showAll(String info) {

        return homeMapper.selectAllByInfo(info);
    }

    @Override
    public List<HanFu> showByLabel(String label) {
        return homeMapper.selectHanFuListByLabel(label);

    }


}
