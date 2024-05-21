package me.iaksh.service;

import me.iaksh.entity.Sales;
import me.iaksh.mapper.SalesMapper;

import java.util.List;

public class SalesService {
    SalesMapper mapper;

    public void setMapper(SalesMapper mapper) {
        this.mapper = mapper;
    }

    public Sales getById(long id) {
        return mapper.getById(id);
    }

    public List<Sales> getAll() {
        return mapper.getAll();
    }

    public void insert(Sales sales) {
        mapper.insert(sales);
    }

    public void update(Sales sales) {
        mapper.update(sales);
    }

    public void deleteById(long id) {
        mapper.deleteById(id);
    }
}
