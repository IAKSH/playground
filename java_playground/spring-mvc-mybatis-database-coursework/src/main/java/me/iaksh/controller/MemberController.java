package me.iaksh.controller;

import com.alibaba.fastjson2.JSON;
import me.iaksh.entity.Member;
import me.iaksh.service.MemberService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/member")
public class MemberController {

    private static MemberService service;

    public static void setService(MemberService service) {
        MemberController.service = service;
    }

    @PostMapping(value = "/update",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String updateMember(@RequestBody String strMember) {
        Member member = JSON.parseObject(strMember,Member.class);
        service.update(member);
        return JSON.toJSONString(member);
    }

    @PostMapping(value = "/insert",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String insertMember(@RequestBody String strMember) {
        Member member = JSON.parseObject(strMember,Member.class);
        service.insert(member);
        return JSON.toJSONString(member);
    }

    @GetMapping(value = "/{id}",
            produces = "application/json;charset=UTF-8")
    public String getMemberById(@PathVariable Long id) {
        return JSON.toJSONString(service.getById(id));
    }

    @GetMapping(value = "/all",
            produces = "application/json;charset=UTF-8")
    public String getAllMembers() {
        return JSON.toJSONString(service.getAll());
    }

    @GetMapping(value = "/delete/{id}",
            produces = "application/json;charset=UTF-8")
    public String deleteMemberById(@PathVariable Long id) {
        service.deleteById(id);
        return "{\n\"status\":\"ok\"\n}";
    }
}