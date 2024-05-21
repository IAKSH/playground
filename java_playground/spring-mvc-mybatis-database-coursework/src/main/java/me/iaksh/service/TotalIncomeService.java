package me.iaksh.service;

import me.iaksh.entity.TotalIncome;
import me.iaksh.mapper.TotalIncomeMapper;

public class TotalIncomeService {
    TotalIncomeMapper mapper;

    public void setMapper(TotalIncomeMapper mapper) {
        this.mapper = mapper;
    }

    public TotalIncome get() {
        return mapper.getIncome().get(0);
    }
}
