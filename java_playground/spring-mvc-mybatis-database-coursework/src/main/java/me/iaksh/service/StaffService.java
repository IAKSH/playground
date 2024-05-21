package me.iaksh.service;

import me.iaksh.entity.Staff;
import me.iaksh.mapper.StaffMapper;

import java.util.List;

public class StaffService {
    StaffMapper mapper;

    public void setMapper(StaffMapper mapper) {
        this.mapper = mapper;
    }

    public Staff getById(long id) {
        return mapper.getById(id);
    }

    public List<Staff> getAll() {
        return mapper.getAll();
    }

    public void insert(Staff staff) {
        mapper.insert(staff);
    }

    public void update(Staff staff) {
        mapper.update(staff);
    }

    public void deleteById(long id) {
        mapper.deleteById(id);
    }
}
