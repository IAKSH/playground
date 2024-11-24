package com.ysc.pojo;


import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class HanFuImage {
    private int imageId;
    private int hanFuId;
    private String image;
}
