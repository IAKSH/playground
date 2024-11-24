package com.ysc.pojo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Act {
    private int actId;
    private String actImage;
    private String actName;
    private String actIntroduce;
    private String actAddress;
    private String actTime;
}
