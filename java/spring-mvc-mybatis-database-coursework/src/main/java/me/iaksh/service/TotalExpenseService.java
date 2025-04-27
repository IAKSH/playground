package me.iaksh.service;

import me.iaksh.entity.TotalExpense;
import me.iaksh.mapper.TotalExpenseMapper;

public class TotalExpenseService {
    TotalExpenseMapper mapper;

    public void setMapper(TotalExpenseMapper mapper) {
        this.mapper = mapper;
    }

    public TotalExpense get() {
        return mapper.getExpense().get(0);
    }
}
