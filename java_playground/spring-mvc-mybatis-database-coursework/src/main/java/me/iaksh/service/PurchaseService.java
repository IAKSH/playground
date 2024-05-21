package me.iaksh.service;

import me.iaksh.entity.Purchase;
import me.iaksh.mapper.PurchaseMapper;

import java.util.List;

public class PurchaseService {
    PurchaseMapper mapper;

    public void setMapper(PurchaseMapper mapper) {
        this.mapper = mapper;
    }

    public Purchase getById(long id) {
        return mapper.getById(id);
    }

    public List<Purchase> getAll() {
        return mapper.getAll();
    }

    public void insert(Purchase purchase) {
        mapper.insert(purchase);
    }

    public void update(Purchase purchase) {
        mapper.update(purchase);
    }

    public void deleteById(long id) {
        mapper.deleteById(id);
    }
}
