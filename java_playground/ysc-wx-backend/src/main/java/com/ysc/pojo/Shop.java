package com.ysc.pojo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class Shop {
    private Integer shopId;
    private String name;
    private float longitude;
    private float latitude;
    private String address;
}
