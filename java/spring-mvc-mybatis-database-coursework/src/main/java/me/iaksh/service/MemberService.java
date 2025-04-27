package me.iaksh.service;

import me.iaksh.entity.Member;
import me.iaksh.mapper.MemberMapper;

import java.util.List;

public class MemberService {
    private MemberMapper mapper;

    public void setMapper(MemberMapper mapper) {
        this.mapper = mapper;
    }

    public Member getById(long id) {
        return mapper.getById(id);
    }

    public List<Member> getAll() {
        return mapper.getAll();
    }

    public void insert(Member menber) {
        mapper.insert(menber);
    }

    public void update(Member menber) {
        mapper.update(menber);
    }

    public void deleteById(long id) {
        mapper.deleteById(id);
    }
}
