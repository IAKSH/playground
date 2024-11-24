package com.ysc.pojo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class User {
    private int userId;
    private String account;
    private String password;
    private String name;
    private String gender;
    private String avatar;
    private String telephone;
    private LocalDateTime createTime;
    private int shopId;
}
