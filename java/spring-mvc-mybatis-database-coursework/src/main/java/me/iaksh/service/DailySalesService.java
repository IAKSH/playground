package me.iaksh.service;


import me.iaksh.entity.DailySales;
import me.iaksh.mapper.DailySalesMapper;

public class DailySalesService {
    private DailySalesMapper mapper;
    public void setMapper(DailySalesMapper mapper) {
        this.mapper = mapper;
    }

    public DailySales get() {
        return mapper.get().get(0);
    }
}
