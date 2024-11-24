package com.ysc.service.impl;

import com.ysc.mapper.ActMapper;
import com.ysc.pojo.Act;
import com.ysc.service.ActService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ActServiceImpl implements ActService {
    @Autowired
    private ActMapper actMapper;
    @Override
    public List<Act> showAllActList() {
        return actMapper.selectAllActList();
    }

    @Override
    public Act showActDetail(int actId) {
        return actMapper.selectActById(actId);
    }
}
