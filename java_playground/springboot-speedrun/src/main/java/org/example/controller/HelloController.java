package org.example.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
public class HelloController {

    @Value("${hello.i}")
    private int i;

    @RequestMapping("/hello")
    String sayHello() {
        return "Hello World! " + (++i);
    }

    @RequestMapping("/hello/{name}")
    String sayHello(@PathVariable String name) {
        return "Hello " + name + "!";
    }

    @GetMapping(value = "/json/{id}",
            produces = "application/json;charset=UTF-8")
    public String testHttpGet(@PathVariable Long id) throws JsonProcessingException {
        Map<String,Object> map=new HashMap<>();
        map.put("name","aihao");
        map.put("age",23);
        map.put("gender","男");
        map.put("received_id",id);
        ObjectMapper mapper = new ObjectMapper();
        return mapper.writeValueAsString(map);
    }

    @PostMapping(value = "/json/post",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String testHttpPost(@RequestBody String strStaff) throws JsonProcessingException {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(strStaff, Map.class).toString();
    }

    @PutMapping(value = "/json/put",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String testHttpPut(@RequestBody String strStaff) throws JsonProcessingException {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(strStaff, Map.class).toString();
    }

    @DeleteMapping(value = "/json/delete/{id}",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String testHttpDelete(@RequestBody String strStaff) throws JsonProcessingException {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(strStaff, Map.class).toString();
    }
}