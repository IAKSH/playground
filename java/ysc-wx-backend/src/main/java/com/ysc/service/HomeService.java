package com.ysc.service;

import com.ysc.pojo.HanFu;
import com.ysc.pojo.HanFuImage;
import com.ysc.pojo.Shop;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public interface HomeService {
    List<HanFu> hanFuAllShow();

    Shop shopNameShow(Integer shopId);

    List<HanFuImage> hanFuImageShow(Integer hanFuId);

    HanFu hanFuShow(Integer hanfuId);

    List<HanFu> showAll(String info);

    List<HanFu> showByLabel(String label);
}
