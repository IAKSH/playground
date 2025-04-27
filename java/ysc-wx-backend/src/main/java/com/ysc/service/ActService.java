package com.ysc.service;

import com.ysc.pojo.Act;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public interface ActService {
    List<Act> showAllActList();

    Act showActDetail(int actId);
}
