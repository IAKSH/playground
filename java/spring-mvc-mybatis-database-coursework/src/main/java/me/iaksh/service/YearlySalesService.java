package me.iaksh.service;

import me.iaksh.entity.YearlySales;
import me.iaksh.mapper.YearlySalesMapper;

public class YearlySalesService {
    YearlySalesMapper mapper;

    public void setMapper(YearlySalesMapper mapper) {
        this.mapper = mapper;
    }

    public YearlySales get() {
        return mapper.get().get(0);
    }
}
