package me.iaksh.controller;

import com.alibaba.fastjson2.JSON;
import me.iaksh.entity.Sales;
import me.iaksh.service.SalesService;
import org.springframework.web.bind.annotation.*;

import java.sql.Timestamp;

@RestController
@RequestMapping("/sales")
public class SalesController {

    private static SalesService service;

    public static void setService(SalesService service) {
        SalesController.service = service;
    }

    @PostMapping(value = "/update",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String updateSales(@RequestBody String strSales) {
        Sales sales = JSON.parseObject(strSales,Sales.class);
        service.update(sales);
        return JSON.toJSONString(sales);
    }

    @PostMapping(value = "/insert",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String insertSales(@RequestBody String strSales) {
        Sales sales = JSON.parseObject(strSales,Sales.class);
        if(sales.getSaleTime() == null) {
            sales.setSaleTime(new Timestamp(System.currentTimeMillis()));
        }
        service.sellProduct(sales);
        return JSON.toJSONString(sales);
    }

    @GetMapping(value = "/{id}",
            produces = "application/json;charset=UTF-8")
    public String getSalesById(@PathVariable Long id) {
        return JSON.toJSONString(service.getById(id));
    }

    @GetMapping(value = "/all",
            produces = "application/json;charset=UTF-8")
    public String getAllSales() {
        return JSON.toJSONString(service.getAll());
    }

    @GetMapping(value = "/delete/{id}",
            produces = "application/json;charset=UTF-8")
    public String deleteSalesById(@PathVariable Long id) {
        service.deleteById(id);
        return "{\n\"status\":\"ok\"\n}";
    }
}
