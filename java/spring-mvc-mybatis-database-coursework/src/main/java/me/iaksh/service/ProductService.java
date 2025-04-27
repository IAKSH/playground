package me.iaksh.service;

import me.iaksh.entity.Product;
import me.iaksh.mapper.ProductMapper;

import java.util.List;

public class ProductService {
    private ProductMapper mapper;

    public void setMapper(ProductMapper mapper) {
        this.mapper = mapper;
    }

    public Product getById(long id) {
        return mapper.getById(id);
    }

    public List<Product> getAll() {
        return mapper.getAll();
    }

    public void insert(Product product) {
        mapper.insert(product);
    }

    public void update(Product product) {
        mapper.update(product);
    }

    public void deleteById(long id) {
        mapper.deleteById(id);
    }
}
