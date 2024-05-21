package me.iaksh.service;

import me.iaksh.entity.Source;
import me.iaksh.mapper.SourceMapper;

import java.util.List;

public class SourceService {
    SourceMapper mapper;

    public void setMapper(SourceMapper mapper) {
        this.mapper = mapper;
    }

    public Source getById(long id) {
        return mapper.getById(id);
    }

    public List<Source> getAll() {
        return mapper.getAll();
    }

    public void insert(Source source) {
        mapper.insert(source);
    }

    public void update(Source source) {
        mapper.update(source);
    }

    public void deleteById(long id) {
        mapper.deleteById(id);
    }
}
