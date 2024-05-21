package me.iaksh.service;

import me.iaksh.entity.MonthlySales;
import me.iaksh.mapper.MonthlySalesMapper;

public class MonthlySalesService {
    private MonthlySalesMapper mapper;

    public void setMapper(MonthlySalesMapper mapper) {
        this.mapper = mapper;
    }

    public MonthlySales get() {
        return mapper.get().get(0);
    }
}
