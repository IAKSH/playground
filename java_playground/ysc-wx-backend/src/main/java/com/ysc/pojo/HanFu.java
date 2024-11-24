package com.ysc.pojo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.sql.Date;
import java.util.List;


@Data
@AllArgsConstructor
@NoArgsConstructor
public class HanFu {
    private int hanFuId;
    private int shopId;
    private String shopName;
    private String name;
    private String price;
    private String image;
    private String label;
    private Date uploadTime;
    private int likes;
    private int views;
    private int cabinetId;
    private List<HanFuImage> imageList;
}
