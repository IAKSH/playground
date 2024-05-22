package me.iaksh.controller;

import com.alibaba.fastjson2.JSON;
import me.iaksh.entity.Staff;
import me.iaksh.service.StaffService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/staff")
public class StaffController {
    private static StaffService service;

    public static void setService(StaffService service) {
        StaffController.service = service;
    }

    @PostMapping(value = "/update",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String updateStaff(@RequestBody String staff) {
        service.update(JSON.parseObject(staff,Staff.class));
        return "{\n\"status\":\"ok\"\n}";
    }

    @PostMapping(value = "/insert",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String insertStaff(@RequestBody String staff) {
        service.insert(JSON.parseObject(staff,Staff.class));
        return "{\n\"status\":\"ok\"\n}";
    }

    @GetMapping(value = "/{id}",
            produces = "application/json;charset=UTF-8")
    public String getStaffById(@PathVariable Long id) {
        return JSON.toJSONString(service.getById(id));
    }

    @GetMapping(value = "/all",
            produces = "application/json;charset=UTF-8")
    public String getAllStaffs() {
        return JSON.toJSONString(service.getAll());
    }

    @GetMapping(value = "/delete/{id}",
            produces = "application/json;charset=UTF-8")
    public String deleteStaffById(@PathVariable Long id) {
        service.deleteById(id);
        return "{\n\"status\":\"ok\"\n}";
    }
}
