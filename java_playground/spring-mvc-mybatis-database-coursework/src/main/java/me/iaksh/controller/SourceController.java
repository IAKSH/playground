package me.iaksh.controller;

import com.alibaba.fastjson2.JSON;
import me.iaksh.entity.Source;
import me.iaksh.service.SourceService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/source")
public class SourceController {
    private static SourceService service;

    public static void setService(SourceService service) {
        SourceController.service = service;
    }

    @PostMapping(value = "/update",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String updateSource(@RequestBody String strSource) {
        Source source = JSON.parseObject(strSource,Source.class);
        service.update(source);
        return JSON.toJSONString(source);
    }

    @PostMapping(value = "/insert",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String insertSource(@RequestBody String strSource) {
        Source source = JSON.parseObject(strSource,Source.class);
        service.insert(source);
        return JSON.toJSONString(source);
    }

    @GetMapping(value = "/{id}",
            produces = "application/json;charset=UTF-8")
    public String getSourceById(@PathVariable Long id) {
        return JSON.toJSONString(service.getById(id));
    }

    @GetMapping(value = "/all",
            produces = "application/json;charset=UTF-8")
    public String getAllSources() {
        return JSON.toJSONString(service.getAll());
    }

    @GetMapping(value = "/delete/{id}",
            produces = "application/json;charset=UTF-8")
    public String deleteSourceById(@PathVariable Long id) {
        service.deleteById(id);
        return "{\n\"status\":\"ok\"\n}";
    }
}
