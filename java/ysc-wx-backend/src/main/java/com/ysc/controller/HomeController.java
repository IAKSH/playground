package com.ysc.controller;

import com.ysc.pojo.HanFu;
import com.ysc.pojo.HanFuImage;
import com.ysc.pojo.Result;
import com.ysc.service.HomeService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/home")
@Slf4j
public class HomeController {
    @Autowired
    private HomeService homeService;

    @GetMapping
    public Result hanFuList(){
        List<HanFu> list=homeService.hanFuAllShow();
        return Result.success(list);
    }

    @GetMapping("/hanfu_detail/{hanfu_id}")
    public Result hanFuImageList(@PathVariable Integer hanfu_id){
        HanFu hanFuInfo=homeService.hanFuShow(hanfu_id);
        List<HanFuImage> imageList =homeService.hanFuImageShow(hanfu_id);
        hanFuInfo.setImageList(imageList);
        return Result.success(hanFuInfo);
    }

    @GetMapping("/search")
    public Result searchHanFuList(String info){
        List<HanFu> list=homeService.showAll(info);
        return Result.success(list);
    }

    @GetMapping("/nav/hanfu_list")
    public Result getHanFuClassifyList(String label){
        List<HanFu> list=homeService.showByLabel(label);
        return Result.success(list);
    }


}
